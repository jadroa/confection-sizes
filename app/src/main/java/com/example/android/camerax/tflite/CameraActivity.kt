/*
 * Copyright 2020 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.example.android.camerax.tflite

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.os.Handler
import android.util.Log
import android.util.Size
import android.view.View
import android.view.ViewGroup
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.constraintlayout.widget.ConstraintLayout
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.Observer
import com.android.example.camerax.tflite.R
import com.example.android.camera.utils.YuvToRgbConverter
import com.example.android.camerax.tflite.imagesegmentation.ImageSegmentationModelExecutor
import kotlinx.android.synthetic.main.activity_camera.*
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.examples.posenet.lib.BodyPart
import org.tensorflow.lite.examples.posenet.lib.Posenet
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import java.util.concurrent.Executors
import kotlin.math.*
import kotlin.random.Random

const val MODEL_WIDTH = 257
const val MODEL_HEIGHT = 257

/** Activity that displays the camera and performs object detection on the incoming frames */
class CameraActivity : AppCompatActivity() {

    private lateinit var container: ConstraintLayout
    private lateinit var bitmapBuffer: Bitmap

    private val executor = Executors.newSingleThreadExecutor()
    private val permissions = listOf(Manifest.permission.CAMERA)
    private val permissionsRequestCode = Random.nextInt(0, 10000)

    private var lensFacing: Int = CameraSelector.LENS_FACING_BACK
    private val isFrontFacing get() = lensFacing == CameraSelector.LENS_FACING_FRONT

    private var pauseAnalysis = false
    private var imageRotationDegrees: Int = 0
    private val tfImageBuffer = TensorImage(DataType.UINT8)

    lateinit var drawOverlayView: DrawOverlayView
    private var imageSegmentationModel: ImageSegmentationModelExecutor? = null
    private val maskImageLiveData: MutableLiveData<SegmentData> = MutableLiveData()
    private val minConfidence = 0.5
    private val bodyJoints = listOf(
            Pair(BodyPart.LEFT_EYE, BodyPart.RIGHT_EYE),
            Pair(BodyPart.LEFT_WRIST, BodyPart.LEFT_ELBOW),
            Pair(BodyPart.LEFT_ELBOW, BodyPart.LEFT_SHOULDER),
            Pair(BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER),
            Pair(BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_ELBOW),
            Pair(BodyPart.RIGHT_ELBOW, BodyPart.RIGHT_WRIST),
            Pair(BodyPart.LEFT_SHOULDER, BodyPart.LEFT_HIP),
            Pair(BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP),
            Pair(BodyPart.RIGHT_HIP, BodyPart.RIGHT_SHOULDER),
            Pair(BodyPart.LEFT_HIP, BodyPart.LEFT_KNEE),
            Pair(BodyPart.LEFT_KNEE, BodyPart.LEFT_ANKLE),
            Pair(BodyPart.RIGHT_HIP, BodyPart.RIGHT_KNEE),
            Pair(BodyPart.RIGHT_KNEE, BodyPart.RIGHT_ANKLE)
    )
    private var prolongedShoulder: Pair<PointF, PointF>? = null
    private var prolongedHip: Pair<PointF, PointF>? = null
    private var lastMeasuredLeftShoulderY: Float? = null
    private var lastMeasuredLeftHipY: Float? = null

    private val tfImageProcessor by lazy {
        val cropSize = minOf(bitmapBuffer.width, bitmapBuffer.height)
        ImageProcessor.Builder()
            .add(ResizeWithCropOrPadOp(cropSize, cropSize))
            .add(ResizeOp(
                tfInputSize.height, tfInputSize.width, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
            .add(Rot90Op(-imageRotationDegrees / 90))
            .add(NormalizeOp(0f, 1f))
            .build()
    }

    private val tflite by lazy {
        Interpreter(
            FileUtil.loadMappedFile(this, MODEL_PATH),
            Interpreter.Options().addDelegate(NnApiDelegate()))
    }

    private val detector by lazy {
        ObjectDetectionHelper(
            tflite,
            FileUtil.loadLabels(this, LABELS_PATH)
        )
    }

    private val tfInputSize by lazy {
        val inputIndex = 0
        val inputShape = tflite.getInputTensor(inputIndex).shape()
        Size(inputShape[2], inputShape[1]) // Order of axis is: {1, height, width, 3}
    }

    private val posenet by lazy {
        Posenet(this)
    }

    override fun onDestroy() {
        super.onDestroy()
        posenet.close()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_camera)
        container = findViewById(R.id.camera_container)

        camera_capture_button.setOnClickListener {

//            // Disable all camera controls
//            it.isEnabled = false
//
//            if (pauseAnalysis) {
//                // If image analysis is in paused state, resume it
//                pauseAnalysis = false
//                image_predicted.visibility = View.GONE
//
//            } else {
//                // Otherwise, pause image analysis and freeze image
//                pauseAnalysis = true
//                val matrix = Matrix().apply {
//                    postRotate(imageRotationDegrees.toFloat())
//                    if (isFrontFacing) postScale(-1f, 1f)
//                }
//                val uprightImage = Bitmap.createBitmap(
//                    bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height, matrix, true)
//                image_predicted.setImageBitmap(uprightImage)
//                image_predicted.visibility = View.VISIBLE
//            }
//
//            // Re-enable camera controls
//            it.isEnabled = true

            drawOverlayView.measureDirection = MeasureDirection.FRONT
            Handler().postDelayed({
                maskImageLiveData.value?.let { segmentData ->
                    drawOverlayView.drawOverlay(segmentData, MeasureDirection.FRONT, MeasuredBodyPart.SHOULDER)?.let { measureShoulderBitmap ->
                        measure(measureShoulderBitmap, MeasuredBodyPart.SHOULDER, MeasureDirection.FRONT)
                        lastMeasuredLeftShoulderY = segmentData.shoulderLine?.second?.y
                    }
                }
                maskImageLiveData.value?.let { segmentData ->
                    drawOverlayView.drawOverlay(segmentData, MeasureDirection.FRONT, MeasuredBodyPart.HIP)?.let { measureHipBitmap ->
                        measure(measureHipBitmap, MeasuredBodyPart.HIP, MeasureDirection.FRONT)
                        lastMeasuredLeftHipY = segmentData.hipLine?.second?.y
                    }
                }
                drawOverlayView.measureDirection = MeasureDirection.SIDE
            }, 10000)

            Handler().postDelayed({
                maskImageLiveData.value?.let { segmentData ->
                    val segmentDataNew = segmentData.copy(
                            shoulderLine = Pair(
                                    PointF(segmentData.shoulderLine?.first?.x!!, lastMeasuredLeftShoulderY!!),
                                    PointF(segmentData.shoulderLine.second.x, lastMeasuredLeftShoulderY!!))
                    )
                    drawOverlayView.drawOverlay(segmentDataNew, MeasureDirection.SIDE, MeasuredBodyPart.SHOULDER)?.let { measureShoulderBitmap ->
                        measure(measureShoulderBitmap, MeasuredBodyPart.SHOULDER, MeasureDirection.SIDE)
                    }
                }
                maskImageLiveData.value?.let { segmentData ->
                    val segmentDataNew = segmentData.copy(
                            hipLine = Pair(
                                    PointF(segmentData.hipLine?.first?.x!!, lastMeasuredLeftHipY!!),
                                    PointF(segmentData.hipLine.second.x, lastMeasuredLeftHipY!!))
                    )
                    drawOverlayView.drawOverlay(segmentDataNew, MeasureDirection.SIDE, MeasuredBodyPart.HIP)?.let { measureHipBitmap ->
                        measure(measureHipBitmap, MeasuredBodyPart.HIP, MeasureDirection.SIDE)
                    }
                }
            }, 20000)

            Handler().postDelayed({
                lastMeasuredLeftHipY = null
                lastMeasuredLeftShoulderY = null
            }, 25000)

        }

        drawOverlayView = findViewById(R.id.draw_overlay)
        maskImageLiveData.observe(this, Observer { segment ->
            if (lastMeasuredLeftShoulderY != null && lastMeasuredLeftHipY != null) {
                drawOverlayView.segment = segment.copy(
                        shoulderLine = Pair(
                                PointF(segment.shoulderLine?.first?.x!!, lastMeasuredLeftShoulderY!!),
                                PointF(segment.shoulderLine.second.x, lastMeasuredLeftShoulderY!!)),
                        hipLine = Pair(
                                PointF(segment.hipLine?.first?.x!!, lastMeasuredLeftHipY!!),
                                PointF(segment.hipLine.second.x, lastMeasuredLeftHipY!!))
                )
            } else {
                drawOverlayView.segment = segment
            }
            drawOverlayView.invalidate()
        })
        createImageSgmentationModelExecutor(false)
    }

    /** Declare and bind preview and analysis use cases */
    @SuppressLint("UnsafeExperimentalUsageError")
    private fun bindCameraUseCases() = view_finder.post {

        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener(Runnable {

            // Camera provider is now guaranteed to be available
            val cameraProvider = cameraProviderFuture.get()

            // Set up the view finder use case to display camera preview
             val preview = Preview.Builder()
                    .setTargetResolution(Size(480, 640))
                .setTargetRotation(view_finder.display.rotation)
                .build()

            // Set up the image analysis use case which will process frames in real time
            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setTargetRotation(view_finder.display.rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            var frameCounter = 0
            var lastFpsTimestamp = System.currentTimeMillis()
            val converter = YuvToRgbConverter(this)

            imageAnalysis.setAnalyzer(executor, ImageAnalysis.Analyzer { image ->
                if (!::bitmapBuffer.isInitialized) {
                    // The image rotation and RGB image buffer are initialized only once
                    // the analyzer has started running
                    imageRotationDegrees = image.imageInfo.rotationDegrees
                    bitmapBuffer = Bitmap.createBitmap(
                        image.width, image.height, Bitmap.Config.ARGB_8888)
                }

                // Early exit: image analysis is in paused state
                if (pauseAnalysis) {
                    image.close()
                    return@Analyzer
                }

                // Convert the image to RGB and place it in our shared buffer
                image.use { converter.yuvToRgb(image.image!!, bitmapBuffer) }

                // Process the image in Tensorflow
//                val tfImage =  tfImageProcessor.process(tfImageBuffer.apply { load(bitmapBuffer) })
//
//                // Perform the object detection for the current frame
//                val predictions = detector.predict(tfImage)
//
//                // Report only the top prediction
//                reportPrediction(predictions.maxBy { it.score })
//
//                // Compute the FPS of the entire pipeline
//                val frameCount = 10
//                if (++frameCounter % frameCount == 0) {
//                    frameCounter = 0
//                    val now = System.currentTimeMillis()
//                    val delta = now - lastFpsTimestamp
//                    val fps = 1000 * frameCount.toFloat() / delta
//                    Log.d(TAG, "FPS: ${"%.02f".format(fps)}")
//                    lastFpsTimestamp = now
//                }

                val rotateMatrix = Matrix()
                rotateMatrix.postRotate(90.0f)
                val rotatedBitmap = Bitmap.createBitmap(
                        bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height,
                        rotateMatrix, true
                )

                val heightRatio = rotatedBitmap.height.toFloat() / MODEL_HEIGHT
                val scaledWidth = (rotatedBitmap.width.toFloat() / heightRatio).toInt()
                val scaledHeight = MODEL_HEIGHT

                val scaledBitmap = Bitmap.createScaledBitmap(rotatedBitmap, scaledWidth, scaledHeight, true)

                val bitmapWithBgrnd = Bitmap.createBitmap(MODEL_WIDTH, MODEL_HEIGHT, Bitmap.Config.ARGB_8888)

                val canvas = Canvas(bitmapWithBgrnd)
                val paint = Paint()
                val offset = ((MODEL_WIDTH / 2) - (scaledWidth / 2)).toFloat()
                canvas.drawBitmap(scaledBitmap, offset, 0f, paint)

                val person = posenet.estimateSinglePose(bitmapWithBgrnd)

                val result = imageSegmentationModel?.execute(bitmapWithBgrnd)
                val maskBitmap = result?.bitmapMaskOnly

                for(line in bodyJoints) {
                    if (
                            (person.keyPoints[line.first.ordinal].score > minConfidence) and
                            (person.keyPoints[line.second.ordinal].score > minConfidence)
                    ) {
                        when (line) {
                            Pair(BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER) -> {
                                val firstX = person.keyPoints[line.first.ordinal].position.x.toFloat() * drawOverlayView.width / MODEL_WIDTH - ((drawOverlayView.width / MODEL_WIDTH) * offset)
                                val firstY = person.keyPoints[line.first.ordinal].position.y.toFloat() * drawOverlayView.width / MODEL_WIDTH
                                val secondX = person.keyPoints[line.second.ordinal].position.x.toFloat() * drawOverlayView.width / MODEL_WIDTH - ((drawOverlayView.width / MODEL_WIDTH) * offset)
                                val secondY = person.keyPoints[line.second.ordinal].position.y.toFloat() * drawOverlayView.width / MODEL_WIDTH
                                prolongedShoulder = prolongLinePoints(Pair(PointF(firstX, firstY), PointF(secondX, secondY)))
                            }
                            Pair(BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP) -> {
                                val firstX = person.keyPoints[line.first.ordinal].position.x.toFloat() * drawOverlayView.width / MODEL_WIDTH - ((drawOverlayView.width / MODEL_WIDTH) * offset)
                                val firstY = person.keyPoints[line.first.ordinal].position.y.toFloat() * drawOverlayView.width / MODEL_WIDTH
                                val secondX = person.keyPoints[line.second.ordinal].position.x.toFloat() * drawOverlayView.width / MODEL_WIDTH - ((drawOverlayView.width / MODEL_WIDTH) * offset)
                                val secondY = person.keyPoints[line.second.ordinal].position.y.toFloat() * drawOverlayView.width / MODEL_WIDTH
                                prolongedHip = prolongLinePoints(Pair(PointF(firstX, firstY), PointF(secondX, secondY)))
                            }
                        }
                        if (prolongedShoulder != null && prolongedHip != null) {
                            maskBitmap?.let {
                                val segmentData = SegmentData(it, prolongedShoulder, prolongedHip)
                                val scaledSegmentData = scaleSegmentData(segmentData)
                                maskImageLiveData.postValue(scaledSegmentData)
                                prolongedShoulder = null
                                prolongedHip = null
                            }
                        }
                    }
                }
            })

            // Create a new camera selector each time, enforcing lens facing
            val cameraSelector = CameraSelector.Builder().requireLensFacing(lensFacing).build()

            // Apply declared configs to CameraX using the same lifecycle owner
            cameraProvider.unbindAll()
            val camera = cameraProvider.bindToLifecycle(
                this as LifecycleOwner, cameraSelector, preview, imageAnalysis)

            // Use the camera object to link our preview use case with the view
            preview.setSurfaceProvider(view_finder.surfaceProvider)

        }, ContextCompat.getMainExecutor(this))
    }

    private fun scaleSegmentData(segment: SegmentData): SegmentData {
        val croppedWidth = (257f * (480f / 640f)).toInt()
        val croppedBitmap = Bitmap.createBitmap(segment.bitmap, (segment.bitmap.width / 2) - (croppedWidth / 2), 0, (257f * (480f / 640f)).toInt(), 257)
        val scaledShoulderLine = Pair(
                PointF(segment.shoulderLine!!.first.x * (640f / 480f), segment.shoulderLine.first.y * (640f / 480f)),
                PointF(segment.shoulderLine.second.x * (640f / 480f), segment.shoulderLine.second.y * (640f / 480f))
        )
        val scaledHipLine = Pair(
                PointF(segment.hipLine!!.first.x * (640f / 480f), segment.hipLine.first.y * (640f / 480f)),
                PointF(segment.hipLine.second.x * (640f / 480f), segment.hipLine.second.y * (640f / 480f))
        )
        return SegmentData(croppedBitmap, scaledShoulderLine, scaledHipLine)
    }

    private fun prolongLinePoints(posePoints: Pair<PointF, PointF>): Pair<PointF, PointF> {
        val prolongedFirstX = posePoints.first.x + 200
        val prolongedFirstY = posePoints.first.y - ((posePoints.second.y - posePoints.first.y) / (posePoints.first.x - posePoints.second.x)) * 200
        val prolongedSecondX = posePoints.second.x - 200
        val prolongedSecondY = posePoints.second.y + ((posePoints.second.y - posePoints.first.y) / (posePoints.first.x - posePoints.second.x)) * 200
        return Pair(PointF(prolongedFirstX, prolongedFirstY), PointF(prolongedSecondX, prolongedSecondY))
    }

    private fun createImageSgmentationModelExecutor(useGPU: Boolean) {
        if (imageSegmentationModel != null) {
            imageSegmentationModel!!.close()
            imageSegmentationModel = null
        }
        try {
            imageSegmentationModel = ImageSegmentationModelExecutor(this, useGPU)
        } catch (e: Exception) {
            Log.e(TAG, "Fail to create ImageSegmentationModelExecutor: ${e.message}")
        }
    }

    private fun reportPrediction(
        prediction: ObjectDetectionHelper.ObjectPrediction?
    ) = view_finder.post {

        // Early exit: if prediction is not good enough, don't report it
        if (prediction == null || prediction.score < ACCURACY_THRESHOLD) {
            box_prediction.visibility = View.GONE
            text_prediction.visibility = View.GONE
            return@post
        }

        // Location has to be mapped to our local coordinates
        val location = mapOutputCoordinates(prediction.location)

        // Update the text and UI
        text_prediction.text = "${"%.2f".format(prediction.score)} ${prediction.label}"
        (box_prediction.layoutParams as ViewGroup.MarginLayoutParams).apply {
            topMargin = location.top.toInt()
            leftMargin = location.left.toInt()
            width = min(view_finder.width, location.right.toInt() - location.left.toInt())
            height = min(view_finder.height, location.bottom.toInt() - location.top.toInt())
        }

        // Make sure all UI elements are visible
        box_prediction.visibility = View.VISIBLE
        text_prediction.visibility = View.VISIBLE
    }

    /**
     * Helper function used to map the coordinates for objects coming out of
     * the model into the coordinates that the user sees on the screen.
     */
    private fun mapOutputCoordinates(location: RectF): RectF {

        // Step 1: map location to the preview coordinates
        val previewLocation = RectF(
            location.left * view_finder.width,
            location.top * view_finder.height,
            location.right * view_finder.width,
            location.bottom * view_finder.height
        )

        // Step 2: compensate for camera sensor orientation and mirroring
        val isFrontFacing = lensFacing == CameraSelector.LENS_FACING_FRONT
        val correctedLocation = if (isFrontFacing) {
            RectF(
                view_finder.width - previewLocation.right,
                previewLocation.top,
                view_finder.width - previewLocation.left,
                previewLocation.bottom)
        } else {
            previewLocation
        }

        // Step 3: compensate for 1:1 to 4:3 aspect ratio conversion + small margin
        val margin = 0.1f
        val requestedRatio = 4f / 3f
        val midX = (correctedLocation.left + correctedLocation.right) / 2f
        val midY = (correctedLocation.top + correctedLocation.bottom) / 2f
        return if (view_finder.width < view_finder.height) {
            RectF(
                midX - (1f + margin) * requestedRatio * correctedLocation.width() / 2f,
                midY - (1f - margin) * correctedLocation.height() / 2f,
                midX + (1f + margin) * requestedRatio * correctedLocation.width() / 2f,
                midY + (1f - margin) * correctedLocation.height() / 2f
            )
        } else {
            RectF(
                midX - (1f - margin) * correctedLocation.width() / 2f,
                midY - (1f + margin) * requestedRatio * correctedLocation.height() / 2f,
                midX + (1f - margin) * correctedLocation.width() / 2f,
                midY + (1f + margin) * requestedRatio * correctedLocation.height() / 2f
            )
        }
    }

    override fun onResume() {
        super.onResume()

        // Request permissions each time the app resumes, since they can be revoked at any time
        if (!hasPermissions(this)) {
            ActivityCompat.requestPermissions(
                this, permissions.toTypedArray(), permissionsRequestCode)
        } else {
            bindCameraUseCases()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == permissionsRequestCode && hasPermissions(this)) {
            bindCameraUseCases()
        } else {
            finish() // If we don't have the required permissions, we can't run
        }
    }

    /** Convenience method used to check if all permissions required by this app are granted */
    private fun hasPermissions(context: Context) = permissions.all {
        ContextCompat.checkSelfPermission(context, it) == PackageManager.PERMISSION_GRANTED
    }

    companion object {
        private val TAG = CameraActivity::class.java.simpleName

        private const val ACCURACY_THRESHOLD = 0.5f
        private const val MODEL_PATH = "coco_ssd_mobilenet_v1_1.0_quant.tflite"
        private const val LABELS_PATH = "coco_ssd_mobilenet_v1_1.0_labels.txt"
    }

    var measureShoulderFront: Double? = null
    var measureShoulderSide: Double? = null
    var measureHipFront: Double? = null
    var measureHipSide: Double? = null

    @Synchronized
    fun measure(measureBitmap: Bitmap, measuredBodyPart: MeasuredBodyPart, measureDirection: MeasureDirection) {
        measureBitmap?.let {
            val pixels = IntArray(it.width * it.height)
            it.getPixels(pixels, 0, it.width, 0, 0, it.width, it.height)
            var leftX = it.width
            var rightX = 0
            var leftY = 0
            var rightY = 0
            var topHeight = it.height
            var bottomHeight = 0
            pixels.forEachIndexed { i, color ->
                val alpha = Color.alpha(color)
                val red = Color.red(color)
                val green = Color.green(color)
                val blue = Color.blue(color)
//                    Log.d("ahoj1", "color=$color red=$red green=$green blue=$blue")
                if (color != 0 && red == 255 && green == 0 && blue == 0) {
//                        Log.d("ahoj2", "red=$red green=$green blue=$blue")
                    val y = i / it.width
                    val x = i - (y * it.width)
                    if (x < leftX) {
                        leftX = x
                        leftY = y
                    }
                    if (x > rightX) {
                        rightX = x
                        rightY = y
                    }
                }

                if (color != 0 && alpha != 0 && !(red == 255 && green == 0 && blue == 0)) {
                    val y = i / it.width
                    if (y < topHeight) {
                        topHeight = y
                    }
                    if (y > bottomHeight) {
                        bottomHeight = y
                    }
                }
            }

            val heightRatio = 179f / (bottomHeight - topHeight).toFloat()
            Log.d("ahoj", "heightRatio=$heightRatio, bottomHeight=$bottomHeight, topHeight=$topHeight")
            Log.d("ahoj", "height in px = $topHeight - $bottomHeight")
            when (measureDirection) {
                MeasureDirection.FRONT -> {
                    if (measuredBodyPart == MeasuredBodyPart.SHOULDER) {
                        measureShoulderFront = sqrt((rightX - leftX).absoluteValue.toDouble().pow(2.toDouble()) + (rightY - leftY).absoluteValue.toDouble().pow(2.toDouble())) * heightRatio
                        Log.d("ahoj", "$measuredBodyPart size front = $measureShoulderFront")
                    } else {
                        measureHipFront = sqrt((rightX - leftX).absoluteValue.toDouble().pow(2.toDouble()) + (rightY - leftY).absoluteValue.toDouble().pow(2.toDouble())) * heightRatio
                        Log.d("ahoj", "$measuredBodyPart size front = $measureHipFront")
                    }
                }

                MeasureDirection.SIDE -> {
                    if (measuredBodyPart == MeasuredBodyPart.SHOULDER) {
                        measureShoulderSide = sqrt((rightX - leftX).absoluteValue.toDouble().pow(2.toDouble()) + (rightY - leftY).absoluteValue.toDouble().pow(2.toDouble())) * heightRatio
                        Log.d("ahoj", "$measuredBodyPart size side = $measureShoulderSide")
                    } else {
                        measureHipSide = sqrt((rightX - leftX).absoluteValue.toDouble().pow(2.toDouble()) + (rightY - leftY).absoluteValue.toDouble().pow(2.toDouble())) * heightRatio
                        Log.d("ahoj", "$measuredBodyPart size side = $measureHipSide")
                    }
                }
            }

            when (measuredBodyPart) {
                MeasuredBodyPart.SHOULDER -> {
                    if (measureShoulderFront != null && measureShoulderSide != null) {
                        val circumference = (measureShoulderFront!! + measureShoulderSide!!) / approximationShoulderConst
                        perimeter_shoulder_view.text = "${circumference.roundToInt()} cm"
                        Log.d("ahoj", "shoulder size circumference = $circumference")
                        measureShoulderFront = null
                        measureShoulderSide = null
                    }
                }
                MeasuredBodyPart.HIP -> {
                    if (measureHipFront != null && measureHipSide != null) {
                        val circumference = (measureHipFront!! + measureHipSide!!) / approximationHipConst
                        perimeter_hip_view.text = "${circumference.roundToInt()} cm"
                        Log.d("ahoj", "hip size circumference = $circumference")
                        measureHipFront = null
                        measureHipSide = null
                    }
                }
            }
        }
    }
}
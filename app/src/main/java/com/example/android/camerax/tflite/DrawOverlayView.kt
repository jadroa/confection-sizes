package com.example.android.camerax.tflite

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import androidx.appcompat.widget.AppCompatImageView

data class SegmentData(
    val bitmap: Bitmap,
    val shoulderLine: Pair<PointF, PointF>? = null,
    val hipLine: Pair<PointF, PointF>? = null
)

enum class MeasureDirection {
    FRONT, SIDE
}

enum class MeasuredBodyPart {
    SHOULDER, HIP
}

const val approximationShoulderConst = 0.6f
const val approximationHipConst = 0.7f

class DrawOverlayView(context: Context, attrs: AttributeSet) : AppCompatImageView(context, attrs) {
    var segment: SegmentData? = null
    var measureDirection: MeasureDirection = MeasureDirection.FRONT

    override fun onDraw(canvas: Canvas?) {
        super.onDraw(canvas)
        segment?.let {
            val overlayBitmap = drawOverlay(it, measureDirection)
            setImageBitmap(overlayBitmap)
        }
    }

    fun drawOverlay(segmentData: SegmentData, measureDirection: MeasureDirection, measuredBodyPart: MeasuredBodyPart? = null): Bitmap? {
        val overlayBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val overlayCanvas = Canvas(overlayBitmap!!)

        val paint = Paint()
        paint.color = Color.GREEN
        drawMaskBitmap(segmentData.bitmap, overlayCanvas, paint)
        paint.strokeWidth = 8.0f
        paint.color = Color.RED
        paint.xfermode = PorterDuffXfermode(PorterDuff.Mode.SRC_IN)

        return when (measuredBodyPart) {
            MeasuredBodyPart.SHOULDER -> {
                segmentData.shoulderLine?.let { line ->
                    drawLine(overlayBitmap, overlayCanvas, paint, line, measureDirection)
                }
            }
            MeasuredBodyPart.HIP -> {
                segmentData.hipLine?.let { line ->
                    drawLine(overlayBitmap, overlayCanvas, paint, line, measureDirection)
                }
            }
            else -> {
                segmentData.shoulderLine?.let { line ->
                    drawLine(overlayBitmap, overlayCanvas, paint, line, measureDirection)
                }
                segmentData.hipLine?.let { line ->
                    drawLine(overlayBitmap, overlayCanvas, paint, line, measureDirection)
                }
                overlayBitmap
                removeMask(overlayBitmap)
            }
        }
    }

    private fun drawLine(overlayBitmap: Bitmap, canvas: Canvas, paint: Paint, endPoints: Pair<PointF, PointF>, measureDirection: MeasureDirection): Bitmap = when (measureDirection) {
        MeasureDirection.FRONT -> {
            canvas.drawLine(endPoints.first.x, endPoints.first.y, endPoints.second.x, endPoints.second.y, paint)
            overlayBitmap
        }
        MeasureDirection.SIDE -> {
            canvas.drawLine(0f, endPoints.second.y, width.toFloat(), endPoints.second.y, paint)
            overlayBitmap
        }
    }

    private fun drawMaskBitmap(maskBitmap: Bitmap, canvas: Canvas, paint: Paint) {
        canvas.drawBitmap(
                maskBitmap,
                Rect(0, 0, maskBitmap.width, maskBitmap.height),
                Rect(0, 0, width, height),
                paint
        )
    }

    private fun removeMask(bitmap: Bitmap): Bitmap {
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        val copyPixels = IntArray(width * height)
        pixels.forEachIndexed { i, pixel ->
            val red = Color.red(pixel)
            val green = Color.green(pixel)
            val blue = Color.blue(pixel)
            copyPixels[i] = if (pixel != 0 && red == 255 && green == 0 && blue == 0) pixel else 0
        }
        return Bitmap.createBitmap(copyPixels, width, height, Bitmap.Config.ARGB_8888)
    }

}
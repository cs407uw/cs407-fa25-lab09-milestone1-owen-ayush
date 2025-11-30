package com.cs407.lab09

/**
 * Represents a ball that can move. (No Android UI imports!)
 *
 * Constructor parameters:
 * - backgroundWidth: the width of the background, of type Float
 * - backgroundHeight: the height of the background, of type Float
 * - ballSize: the width/height of the ball, of type Float
 */
class Ball(
    private val backgroundWidth: Float,
    private val backgroundHeight: Float,
    private val ballSize: Float
) {
    var posX = 0f
    var posY = 0f
    var velocityX = 0f
    var velocityY = 0f
    private var accX = 0f
    private var accY = 0f

    private var isFirstUpdate = true

    init {
        reset()
    }

    /**
     * Updates the ball's position and velocity based on the given acceleration and time step.
     * (See lab handout for physics equations)
     */
    fun updatePositionAndVelocity(xAcc: Float, yAcc: Float, dT: Float) {
        if(isFirstUpdate) {
            isFirstUpdate = false
            accX = xAcc
            accY = yAcc
            return
        }

        // Calculate new velocity using equation (1): v1 = v0 + 1/2 * (a1 + a0) * dT
        val newVelocityX = velocityX + 0.5f * (xAcc + accX) * dT
        val newVelocityY = velocityY + 0.5f * (yAcc + accY) * dT

        // Calculate distance traveled using equation (2): l = v0 * dT + 1/6 * dT^2 * (3*a0 + a1)
        val distanceX = velocityX * dT + (1.0f / 6.0f) * dT * dT * (3 * accX + xAcc)
        val distanceY = velocityY * dT + (1.0f / 6.0f) * dT * dT * (3 * accY + yAcc)

        // Update position
        posX += distanceX
        posY += distanceY

        // Update velocity
        velocityX = newVelocityX
        velocityY = newVelocityY

        // Update acceleration for next iteration
        accX = xAcc
        accY = yAcc

        // Check and handle boundaries
        checkBoundaries()
    }

    /**
     * Ensures the ball does not move outside the boundaries.
     * When it collides, velocity and acceleration perpendicular to the
     * boundary should be set to 0.
     */
    fun checkBoundaries() {
        // Check left boundary
        if (posX < 0) {
            posX = 0f
            velocityX = 0f
            accX = 0f
        }

        // Check right boundary
        if (posX > backgroundWidth - ballSize) {
            posX = backgroundWidth - ballSize
            velocityX = 0f
            accX = 0f
        }

        // Check top boundary
        if (posY < 0) {
            posY = 0f
            velocityY = 0f
            accY = 0f
        }

        // Check bottom boundary
        if (posY > backgroundHeight - ballSize) {
            posY = backgroundHeight - ballSize
            velocityY = 0f
            accY = 0f
        }
    }

    /**
     * Resets the ball to the center of the screen with zero
     * velocity and acceleration.
     */
    fun reset() {
        // Reset position to center of the screen
        posX = (backgroundWidth - ballSize) / 2f
        posY = (backgroundHeight - ballSize) / 2f

        // Reset velocity to zero
        velocityX = 0f
        velocityY = 0f

        // Reset acceleration to zero
        accX = 0f
        accY = 0f

        // Reset the first update flag
        isFirstUpdate = true
    }
}
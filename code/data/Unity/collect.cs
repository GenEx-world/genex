using UnityEngine;
using System.IO;

public class CameraMover : MonoBehaviour
{
    public float distance = 20f;      // Total distance to move
    public int totalFrames = 50;      // Total number of frames
    public int outputWidth = 500;     // Output image width
    public int outputHeight = 500;    // Output image height
    public string outputDirectory = "path/to/save/folder";  // Output directory

    public float minX, maxX, minY, maxY, minZ, maxZ;  // Boundaries for random position sampling
    public int maxSamplingAttempts = 100;             // Maximum number of attempts to find a valid path
    public int iterations = 10000;                    // Number of iterations to perform

    private int currentFrame = 0;     // Current frame counter
    private Vector3 startPosition;    // Starting position of the camera
    private Vector3 targetPosition;   // Target position of the camera
    private Vector3 movementDirection;// Movement direction of the camera

    void Start()
    {
        for (int i = 0; i < iterations; i++)
        {
            // Create a new folder for this iteration
            string iterationFolder = Path.Combine(outputDirectory, $"Iteration_{i}");
            if (Directory.Exists(iterationFolder))
            {
                // Skip if folder exists
                continue;
            }
            Directory.CreateDirectory(iterationFolder);

            // Find a valid random start position and orientation
            bool validPathFound = false;
            int attempts = 0;

            while (!validPathFound && attempts < maxSamplingAttempts)
            {
                attempts++;
                SampleRandomPositionAndOrientation();

                // Check if the path is valid
                if (IsPathClear())
                {
                    validPathFound = true;
                }
            }

            if (!validPathFound)
            {
                Debug.LogError($"Unable to find a valid path after multiple attempts for iteration {i}.");
                continue;
            }

            // Perform the movement and capture images for this iteration
            CaptureAndSaveImages(iterationFolder);
        }

        // Ensure all unused assets are unloaded to free up memory
        Resources.UnloadUnusedAssets();
    }

    void SampleRandomPositionAndOrientation()
    {
        // Sample a random position within the specified boundaries
        startPosition = new Vector3(
            Random.Range(minX, maxX),
            Random.Range(minY, maxY),
            Random.Range(minZ, maxZ)
        );

        // Sample a random orientation (rotation around the Y-axis)
        float randomRotationY = Random.Range(0f, 360f);
        movementDirection = Quaternion.Euler(0, randomRotationY, 0) * Vector3.forward;

        // Set the target position based on the sampled orientation and distance
        targetPosition = startPosition + movementDirection * distance;

        // Set the camera's initial position and rotation
        transform.position = startPosition;
        transform.rotation = Quaternion.LookRotation(movementDirection);
    }

    bool IsPathClear()
    {
        // Check if the path between startPosition and targetPosition is clear
        RaycastHit hit;
        if (Physics.Raycast(startPosition, movementDirection, out hit, distance))
        {
            // Collision detected
            return false;
        }

        // No collision detected
        return true;
    }

    void CaptureAndSaveImages(string iterationFolder)
    {
        for (currentFrame = 0; currentFrame < totalFrames; currentFrame++)
        {
            // Move the camera strictly in the forward direction of the current orientation
            transform.position = Vector3.Lerp(startPosition, targetPosition, (float)currentFrame / totalFrames);
            CaptureAndSaveFaces(iterationFolder);
        }

        // Manually trigger garbage collection after each iteration to release unused memory
        System.GC.Collect();
    }

    void CaptureAndSaveFaces(string iterationFolder)
    {
        // Correctly set the camera rotation for each face
        Quaternion originalRotation = transform.rotation;

        string[] faceNames = new string[]
        {
            "back", "left", "front", "right", "top", "bottom"
        };

        Quaternion[] rotations = new Quaternion[]
        {
            Quaternion.LookRotation(-transform.forward),  // Back face
            Quaternion.LookRotation(-transform.right),    // Left face
            Quaternion.LookRotation(transform.forward),   // Front face
            Quaternion.LookRotation(transform.right),     // Right face
            Quaternion.LookRotation(transform.forward) * Quaternion.Euler(-90f, 0, 0),  // Top face, rotated to align with front
            Quaternion.LookRotation(transform.forward) * Quaternion.Euler(90f, 0, 0) // Bottom face, rotated to align with front
        };

        for (int i = 0; i < rotations.Length; i++)
        {
            // Set the camera rotation to the correct direction
            transform.rotation = rotations[i];
            SaveImage(iterationFolder, faceNames[i]);
        }

        // Restore the original rotation after capturing the faces
        transform.rotation = originalRotation;
    }

    void SaveImage(string iterationFolder, string faceName)
    {
        RenderTexture rt = new RenderTexture(outputWidth, outputHeight, 24);
        Camera.main.targetTexture = rt;
        Texture2D screenShot = new Texture2D(outputWidth, outputHeight, TextureFormat.RGB24, false);
        Camera.main.Render();
        RenderTexture.active = rt;
        screenShot.ReadPixels(new Rect(0, 0, outputWidth, outputHeight), 0, 0);
        screenShot.Apply();
        Camera.main.targetTexture = null;
        RenderTexture.active = null;

        // Encode texture to PNG and save it
        string filename = Path.Combine(iterationFolder, $"{currentFrame}_{faceName}.png");
        byte[] bytes = screenShot.EncodeToPNG();
        File.WriteAllBytes(filename, bytes);

        // Properly release memory
        Destroy(rt);
        Destroy(screenShot);
    }
}

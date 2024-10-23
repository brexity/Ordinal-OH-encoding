import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def draw_lines(img, lines):
    if lines is None:
        print("No lines detected")
        return
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 10)

def process_frame(frame):
    height, width = frame.shape[:2]
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to the grayscale frame
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Perform Canny edge detection
    edges = cv2.Canny(blur, 50, 150)
    
    # Define a region of interest (ROI) where the lanes are likely to be located
    roi_vertices = np.array([[(0, height), (width // 2, height // 2), 
                              (width, height)]], dtype=np.int32)
    cropped_edges = region_of_interest(edges, roi_vertices)
    
    # Perform Hough Line Transformation to detect lines
    lines = cv2.HoughLinesP(cropped_edges, rho=1, theta=np.pi/180, threshold=50, 
                            minLineLength=50, maxLineGap=200)
    
    # Create an image to draw lines on
    line_image = np.zeros_like(frame)
    
    # Draw the detected lines on the black image
    draw_lines(line_image, lines)
    
    # Combine the original frame with the line image
    output = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    
    return output

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process the current frame to detect lanes
        processed_frame = process_frame(frame)
        
        # Display the processed frame
        cv2.imshow('Lane Detection', processed_frame)
        
        # Press 'q' to quit the video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(r"Users\Sarthak\Downloads\Lane Detection Test Video 01.mp4")

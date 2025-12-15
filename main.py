import cv2
import pygame
import numpy as np
import sys
from concurrent.futures import ThreadPoolExecutor
import threading

class Tracker:
    def __init__(self, x, y, frame, patch_size=32, search_size=100, auto_created=False):
        self.initial_pos = (x, y)
        self.current_pos = (x, y)
        self.patch_size = patch_size
        self.search_size = search_size
        self.correlation_score = 1.0
        self.reference_patch = self._extract_patch(frame, x, y)
        self.alive = True
        self.lock = threading.Lock()
        self.auto_created = auto_created
        self.frames_lived = 0
    
    def _extract_patch(self, image, x, y):
        h, w = image.shape[:2]
        x1 = max(0, x - self.patch_size)
        y1 = max(0, y - self.patch_size)
        x2 = min(w, x + self.patch_size + 1)
        y2 = min(h, y + self.patch_size + 1)
        return image[y1:y2, x1:x2]
    
    def update(self, frame):
        if not self.alive:
            return
        
        h, w = frame.shape[:2]
        patch_h, patch_w = self.reference_patch.shape[:2]
        
        # Convert to grayscale for correlation
        if len(frame.shape) == 3:
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = frame
        
        if len(self.reference_patch.shape) == 3:
            gray_ref = cv2.cvtColor(self.reference_patch, cv2.COLOR_BGR2GRAY)
        else:
            gray_ref = self.reference_patch
        
        # Search area bounds
        center_x, center_y = self.current_pos
        y_start = max(patch_h//2, center_y - self.search_size)
        y_end = min(h - patch_h//2, center_y + self.search_size + 1)
        x_start = max(patch_w//2, center_x - self.search_size)
        x_end = min(w - patch_w//2, center_x + self.search_size + 1)
        
        # Template matching in search area
        search_area = gray_image[y_start - patch_h//2:y_end + patch_h//2,
                                  x_start - patch_w//2:x_end + patch_w//2]
        
        if search_area.shape[0] >= patch_h and search_area.shape[1] >= patch_w:
            result = cv2.matchTemplate(search_area, gray_ref, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            with self.lock:
                self.correlation_score = max_val
                if max_val >= 0.6:
                    self.current_pos = (x_start + max_loc[0], y_start + max_loc[1])
                    self.frames_lived += 1
                else:
                    self.alive = False
    
    def draw(self, screen, font):
        with self.lock:
            if not self.alive:
                return
            
            current_pos = self.current_pos
            correlation_score = self.correlation_score
            frames_lived = self.frames_lived
        
        # Draw blue rectangle for search area
        search_rect = pygame.Rect(
            current_pos[0] - self.search_size,
            current_pos[1] - self.search_size,
            2 * self.search_size + 1,
            2 * self.search_size + 1
        )
        # Clip to screen bounds
        search_rect.clamp_ip(pygame.Rect(0, 0, screen.get_width(), screen.get_height() - 50))
        pygame.draw.rect(screen, (0, 100, 255), search_rect, 1)
        
        # Draw rectangle at current position (green for manual, cyan for auto)
        patch_rect = pygame.Rect(
            current_pos[0] - self.patch_size,
            current_pos[1] - self.patch_size,
            2 * self.patch_size + 1,
            2 * self.patch_size + 1
        )
        color = (0, 255, 255) if self.auto_created else (0, 255, 0)
        pygame.draw.rect(screen, color, patch_rect, 2)
        
        # Draw red dot at current position
        pygame.draw.circle(screen, (255, 0, 0), current_pos, 5, 2)
        
        # Display correlation score
        corr_text = font.render(f"{correlation_score:.3f}", True, (255, 255, 0))
        text_x = min(current_pos[0] + 10, screen.get_width() - 50)
        text_y = max(0, current_pos[1] - 10)
        screen.blit(corr_text, (text_x, text_y))
        
        # Display frames lived below correlation score
        frames_text = font.render(f"{frames_lived}", True, (255, 255, 255))
        frames_y = min(current_pos[1] + 10, screen.get_height() - 50)
        screen.blit(frames_text, (text_x, frames_y))

pygame.init()

cap = cv2.VideoCapture("VID20251205101021.mp4")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
current_frame = 0

ret, frame = cap.read()
if not ret:
    print("Cannot read video")
    sys.exit()

height, width, _ = frame.shape
screen = pygame.display.set_mode((width, height + 50))
pygame.display.set_caption("Video Player")
clock = pygame.time.Clock()

playing = True
dragging = False
speed_multiplier = 1.0
user_trackers = []
auto_trackers = []
executor = ThreadPoolExecutor(max_workers=8)

def draw_slider(screen, current_frame, total_frames, width):
    slider_rect = pygame.Rect(10, height + 10, width - 20, 20)
    pygame.draw.rect(screen, (100, 100, 100), slider_rect)
    
    if total_frames > 0:
        slider_pos = int((current_frame / (total_frames - 1)) * (width - 20))
        handle_rect = pygame.Rect(slider_pos + 5, height + 10, 10, 20)
        pygame.draw.rect(screen, (255, 255, 255), handle_rect)
    
    return slider_rect

def get_frame_from_slider_pos(mouse_x, width, total_frames):
    relative_x = max(0, min(mouse_x - 10, width - 20))
    return int((relative_x / (width - 20)) * (total_frames - 1))

def find_good_features(frame, user_trackers, auto_trackers, max_trackers=10, min_distance=500):
    """Find good corner features to track, avoiding existing tracker positions"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    
    # Get all existing tracker positions (both user and auto)
    existing_positions = []
    for tracker in user_trackers + auto_trackers:
        with tracker.lock:
            if tracker.alive:
                existing_positions.append(tracker.current_pos)
    
    # Find corners using goodFeaturesToTrack
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=100,  # Find more candidates than we need
        qualityLevel=0.01,
        minDistance=50,  # Minimum distance between detected corners
        blockSize=3
    )
    
    if corners is None:
        return []
    
    # Convert to list of (x, y, quality) tuples
    candidates = []
    for corner in corners:
        x, y = corner.ravel()
        x, y = int(x), int(y)
        
        # Calculate quality score using corner response
        if 32 <= x < gray.shape[1] - 32 and 32 <= y < gray.shape[0] - 32:
            # Base quality measure using local variance
            patch = gray[y-16:y+16, x-16:x+16]
            corner_quality = np.var(patch)
            
            # Distance bonus: farther from existing trackers = higher quality
            min_distance_to_existing = float('inf')
            for ex_x, ex_y in existing_positions:
                distance = np.sqrt((x - ex_x)**2 + (y - ex_y)**2)
                min_distance_to_existing = min(min_distance_to_existing, distance)
            
            # Calculate feature stability - penalize straight edges that can "slide"
            # Check gradient directions in a small window around the point
            if y > 5 and y < gray.shape[0] - 5 and x > 5 and x < gray.shape[1] - 5:
                # Calculate gradients in x and y directions
                gx = cv2.Sobel(gray[y-5:y+5, x-5:x+5], cv2.CV_64F, 1, 0, ksize=3)
                gy = cv2.Sobel(gray[y-5:y+5, x-5:x+5], cv2.CV_64F, 0, 1, ksize=3)
                
                # Calculate gradient magnitudes
                grad_mag = np.sqrt(gx**2 + gy**2)
                
                # Check for corner-like features vs edge-like features
                # Good corners have strong gradients in multiple directions
                gx_var = np.var(gx)
                gy_var = np.var(gy)
                
                # Corner score: high when both x and y gradients have variance (intersecting edges)
                # Low when only one direction has strong gradients (straight edge)
                corner_stability = min(gx_var, gy_var) / (max(gx_var, gy_var) + 1e-6)
                stability_bonus = corner_stability * 500  # Bonus for corner-like features
            else:
                stability_bonus = 0
            
            # Distance from image edges (prefer points away from edges for tracking space)
            edge_distance = min(x, y, gray.shape[1] - x, gray.shape[0] - y)
            
            # If no existing trackers, use base quality with edge and stability consideration
            if min_distance_to_existing == float('inf'):
                final_quality = corner_quality + edge_distance * 2 + stability_bonus
            else:
                # Combine corner quality with distance bonus, edge distance, and stability
                distance_bonus = min_distance_to_existing / 10.0  # Scale distance
                edge_bonus = edge_distance / 10.0  # Reduced edge bonus weight
                final_quality = corner_quality + distance_bonus * 1000 + edge_bonus * 50 + stability_bonus
            
            candidates.append((x, y, final_quality))
    
    # Sort by quality (highest first) - now favors distance
    candidates.sort(key=lambda c: c[2], reverse=True)
    
    # Filter candidates by distance to existing trackers
    good_candidates = []
    for x, y, quality in candidates:
        # Check distance to existing trackers
        too_close = False
        for ex_x, ex_y in existing_positions:
            distance = np.sqrt((x - ex_x)**2 + (y - ex_y)**2)
            if distance < min_distance:
                too_close = True
                break
        
        if not too_close:
            # Check distance to already selected candidates
            for gx, gy, _ in good_candidates:
                distance = np.sqrt((x - gx)**2 + (y - gy)**2)
                if distance < min_distance:
                    too_close = True
                    break
        
        if not too_close:
            good_candidates.append((x, y, quality))
        
        # Stop when we have enough candidates
        if len(good_candidates) >= max_trackers:
            break
    
    return good_candidates

try:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                executor.shutdown(wait=False)
                pygame.quit()
                cap.release()
                sys.exit()
        
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    playing = not playing
                elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    executor.shutdown(wait=False)
                    pygame.quit()
                    cap.release()
                    sys.exit()
                elif event.key == pygame.K_LEFT:
                    current_frame = max(0, current_frame - 1)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                elif event.key == pygame.K_RIGHT:
                    current_frame = min(total_frames - 1, current_frame + 1)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    speed_multiplier = min(16.0, speed_multiplier * 2.0)
                elif event.key == pygame.K_MINUS:
                    speed_multiplier = max(0.125, speed_multiplier / 2.0)
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.pos[1] > height:
                    dragging = True
                    current_frame = get_frame_from_slider_pos(event.pos[0], width, total_frames)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                else:
                    # Create a new user tracker at click position
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                    ret, frame = cap.read()
                    if ret:
                        new_tracker = Tracker(event.pos[0], event.pos[1], frame, auto_created=False)
                        user_trackers.append(new_tracker)
                        print(f"User tracker at: ({event.pos[0]}, {event.pos[1]})")
                        print(f"Total trackers: {len(user_trackers + auto_trackers)}")
            
            elif event.type == pygame.MOUSEBUTTONUP:
                dragging = False
            
            elif event.type == pygame.MOUSEMOTION and dragging:
                current_frame = get_frame_from_slider_pos(event.pos[0], width, total_frames)
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    
        if playing:
            ret, frame = cap.read()
            if ret:
                current_frame += 1
            else:
                # Loop video
                current_frame = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
        
        if ret:
            # Only update trackers when video is playing
            if playing:
                # Update all trackers in parallel
                all_trackers = user_trackers + auto_trackers
                if all_trackers:
                    futures = [executor.submit(tracker.update, frame) for tracker in all_trackers]
                    # Wait for all updates to complete
                    for future in futures:
                        future.result()
                
                # Remove dead trackers
                user_trackers[:] = [t for t in user_trackers if t.alive]
                auto_trackers[:] = [t for t in auto_trackers if t.alive]
                
                # Only search for features when we actually need more auto trackers
                trackers_needed = 10 - len(auto_trackers)
                if trackers_needed > 0:
                    candidates = find_good_features(frame, user_trackers, auto_trackers, max_trackers=trackers_needed, min_distance=300)
                    for x, y, quality in candidates:
                        if len(auto_trackers) >= 10:
                            break
                        new_tracker = Tracker(x, y, frame, auto_created=True)
                        auto_trackers.append(new_tracker)
                        print(f"Auto-created tracker at ({x}, {y}) with quality {quality:.1f}")
            
            # Dim the frame by 50% to make trackers more visible
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = (frame_rgb * 0.5).astype(np.uint8)
            frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
            screen.blit(frame_surface, (0, 0))
        
        screen.fill((50, 50, 50), (0, height, width, 50))
        draw_slider(screen, current_frame, total_frames, width)
        
        font = pygame.font.Font(None, 24)
        frame_text = font.render(f"Frame: {current_frame}/{total_frames}", True, (255, 255, 255))
        speed_text = font.render(f"Speed: {speed_multiplier}x", True, (255, 255, 255))
        tracker_text = font.render(f"User: {len(user_trackers)} Auto: {len(auto_trackers)}", True, (255, 255, 255))
        screen.blit(frame_text, (width - 200, height + 15))
        screen.blit(speed_text, (10, height + 25))
        screen.blit(tracker_text, (width // 2 - 80, height + 15))
        
        # Draw all trackers
        for tracker in user_trackers + auto_trackers:
            tracker.draw(screen, font)
    
        pygame.display.flip()
        clock.tick(fps * speed_multiplier if playing else 60)
        
except KeyboardInterrupt:
    executor.shutdown(wait=False)
    pygame.quit()
    cap.release()
    sys.exit()
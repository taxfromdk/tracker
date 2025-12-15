import cv2
import pygame
import numpy as np
import sys
from concurrent.futures import ThreadPoolExecutor
import threading
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import OpenGL.GL as gl

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
    
    def draw_opengl(self):
        with self.lock:
            if not self.alive:
                return
            
            current_pos = self.current_pos
            correlation_score = self.correlation_score
            frames_lived = self.frames_lived
        
        # Draw using OpenGL immediate mode (can be optimized with VBOs later)
        glDisable(GL_TEXTURE_2D)
        
        # Draw search area rectangle (blue)
        glColor3f(0.0, 0.4, 1.0)
        glLineWidth(1)
        glBegin(GL_LINE_LOOP)
        glVertex2f(current_pos[0] - self.search_size, current_pos[1] - self.search_size)
        glVertex2f(current_pos[0] + self.search_size, current_pos[1] - self.search_size)
        glVertex2f(current_pos[0] + self.search_size, current_pos[1] + self.search_size)
        glVertex2f(current_pos[0] - self.search_size, current_pos[1] + self.search_size)
        glEnd()
        
        # Draw patch rectangle (green for manual, cyan for auto)
        if self.auto_created:
            glColor3f(0.0, 1.0, 1.0)  # Cyan
        else:
            glColor3f(0.0, 1.0, 0.0)  # Green
        
        glLineWidth(2)
        glBegin(GL_LINE_LOOP)
        glVertex2f(current_pos[0] - self.patch_size, current_pos[1] - self.patch_size)
        glVertex2f(current_pos[0] + self.patch_size, current_pos[1] - self.patch_size)
        glVertex2f(current_pos[0] + self.patch_size, current_pos[1] + self.patch_size)
        glVertex2f(current_pos[0] - self.patch_size, current_pos[1] + self.patch_size)
        glEnd()
        
        # Draw red dot
        glColor3f(1.0, 0.0, 0.0)
        glPointSize(5)
        glBegin(GL_POINTS)
        glVertex2f(current_pos[0], current_pos[1])
        glEnd()
        
        # Note: Text rendering in OpenGL requires more setup
        # For now, we'll handle text separately or use bitmap fonts

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
screen = pygame.display.set_mode((width, height + 50), pygame.OPENGL | pygame.DOUBLEBUF)
pygame.display.set_caption("Video Player")
clock = pygame.time.Clock()

# OpenGL setup
glViewport(0, 0, width, height + 50)
glMatrixMode(GL_PROJECTION)
glLoadIdentity()
glOrtho(0, width, height + 50, 0, -1, 1)
glMatrixMode(GL_MODELVIEW)
glLoadIdentity()

# Enable blending for overlay elements
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

# Create texture for video frame
video_texture = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, video_texture)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

# Create 4096x4096 render target
render_texture_size = 4096
render_texture = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, render_texture)
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, render_texture_size, render_texture_size, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

# Create framebuffer for render texture
framebuffer = glGenFramebuffers(1)
glBindFramebuffer(GL_FRAMEBUFFER, framebuffer)
glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, render_texture, 0)

# Check framebuffer completeness
if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
    print("Framebuffer not complete!")

# Return to default framebuffer
glBindFramebuffer(GL_FRAMEBUFFER, 0)

playing = True
dragging = False
speed_multiplier = 1.0
user_trackers = []
auto_trackers = []
executor = ThreadPoolExecutor(max_workers=8)
show_render_texture = False
start_time = pygame.time.get_ticks()

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
                elif event.key == pygame.K_r:
                    show_render_texture = not show_render_texture
                    print(f"Render texture mode: {'ON' if show_render_texture else 'OFF'}")
            
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
                # End of video - reset everything
                print("Video ended - resetting trackers and render texture")
                
                # Clear all trackers
                user_trackers.clear()
                auto_trackers.clear()
                
                # Reset render texture to black
                glBindFramebuffer(GL_FRAMEBUFFER, framebuffer)
                glViewport(0, 0, render_texture_size, render_texture_size)
                glClearColor(0.0, 0.0, 0.0, 1.0)
                glClear(GL_COLOR_BUFFER_BIT)
                glBindFramebuffer(GL_FRAMEBUFFER, 0)
                glViewport(0, 0, width, height + 50)
                
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
            
            # Upload frame to OpenGL texture
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = (frame_rgb * 0.5).astype(np.uint8)  # Dim by 50%
            
            # Flip vertically for OpenGL coordinate system
            frame_rgb = np.flipud(frame_rgb)
            
            glBindTexture(GL_TEXTURE_2D, video_texture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, frame_rgb)
            
            # Calculate rotation angle (one revolution per second)
            current_time = pygame.time.get_ticks()
            rotation_angle = ((current_time - start_time) / 1000.0) * 360.0  # degrees per second
            
            # Render to 4096x4096 texture
            glBindFramebuffer(GL_FRAMEBUFFER, framebuffer)
            glViewport(0, 0, render_texture_size, render_texture_size)
            
            # Set up orthographic projection for render texture
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glOrtho(0, render_texture_size, render_texture_size, 0, -1, 1)
            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glLoadIdentity()
            
            # Don't clear render texture - pixels outside stamp remain persistent
            
            # Draw rotated video at center
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, video_texture)
            glColor3f(1.0, 1.0, 1.0)
            
            # Move to center and rotate
            glTranslatef(render_texture_size/2, render_texture_size/2, 0)
            glRotatef(rotation_angle, 0, 0, 1)
            
            # Scale video to fit nicely in render texture (adjust as needed)
            scale = min(render_texture_size / width, render_texture_size / height) * 0.3
            glScalef(scale, scale, 1)
            
            glBegin(GL_QUADS)
            glTexCoord2f(0, 1); glVertex2f(-width/2, -height/2)
            glTexCoord2f(1, 1); glVertex2f(width/2, -height/2)
            glTexCoord2f(1, 0); glVertex2f(width/2, height/2)
            glTexCoord2f(0, 0); glVertex2f(-width/2, height/2)
            glEnd()
            
            glDisable(GL_TEXTURE_2D)
            
            # Restore matrices
            glPopMatrix()
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            
            # Return to default framebuffer
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            glViewport(0, 0, width, height + 50)
            
            # Set up original projection
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            glOrtho(0, width, height + 50, 0, -1, 1)
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            
            # Clear main screen
            glClear(GL_COLOR_BUFFER_BIT)
            
            if show_render_texture:
                # Show the 4096x4096 render texture
                glEnable(GL_TEXTURE_2D)
                glBindTexture(GL_TEXTURE_2D, render_texture)
                glColor3f(1.0, 1.0, 1.0)
                
                glBegin(GL_QUADS)
                glTexCoord2f(0, 1); glVertex2f(0, 0)
                glTexCoord2f(1, 1); glVertex2f(width, 0)
                glTexCoord2f(1, 0); glVertex2f(width, height)
                glTexCoord2f(0, 0); glVertex2f(0, height)
                glEnd()
                
                glDisable(GL_TEXTURE_2D)
            else:
                # Show normal video
                glEnable(GL_TEXTURE_2D)
                glBindTexture(GL_TEXTURE_2D, video_texture)
                glColor3f(1.0, 1.0, 1.0)
                
                glBegin(GL_QUADS)
                glTexCoord2f(0, 1); glVertex2f(0, 0)
                glTexCoord2f(1, 1); glVertex2f(width, 0)
                glTexCoord2f(1, 0); glVertex2f(width, height)
                glTexCoord2f(0, 0); glVertex2f(0, height)
                glEnd()
                
                glDisable(GL_TEXTURE_2D)
        
        # Draw UI background
        glColor3f(0.2, 0.2, 0.2)
        glBegin(GL_QUADS)
        glVertex2f(0, height)
        glVertex2f(width, height)
        glVertex2f(width, height + 50)
        glVertex2f(0, height + 50)
        glEnd()
        
        # Draw slider background
        glColor3f(0.4, 0.4, 0.4)
        glBegin(GL_QUADS)
        glVertex2f(10, height + 10)
        glVertex2f(width - 10, height + 10)
        glVertex2f(width - 10, height + 30)
        glVertex2f(10, height + 30)
        glEnd()
        
        # Draw slider handle
        if total_frames > 0:
            slider_pos = 10 + int((current_frame / (total_frames - 1)) * (width - 20))
            glColor3f(1.0, 1.0, 1.0)
            glBegin(GL_QUADS)
            glVertex2f(slider_pos, height + 10)
            glVertex2f(slider_pos + 10, height + 10)
            glVertex2f(slider_pos + 10, height + 30)
            glVertex2f(slider_pos, height + 30)
            glEnd()
        
        # Only draw trackers in normal video mode
        if not show_render_texture:
            for tracker in user_trackers + auto_trackers:
                tracker.draw_opengl()
            
            # Draw 600x600 render texture preview in top right corner
            preview_size = 600
            preview_x = width - preview_size - 10
            preview_y = 10
            
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, render_texture)
            glColor3f(1.0, 1.0, 1.0)
            
            glBegin(GL_QUADS)
            glTexCoord2f(0, 1); glVertex2f(preview_x, preview_y)
            glTexCoord2f(1, 1); glVertex2f(preview_x + preview_size, preview_y)
            glTexCoord2f(1, 0); glVertex2f(preview_x + preview_size, preview_y + preview_size)
            glTexCoord2f(0, 0); glVertex2f(preview_x, preview_y + preview_size)
            glEnd()
            
            # Draw border around preview
            glDisable(GL_TEXTURE_2D)
            glColor3f(0.5, 0.5, 0.5)
            glLineWidth(2)
            glBegin(GL_LINE_LOOP)
            glVertex2f(preview_x, preview_y)
            glVertex2f(preview_x + preview_size, preview_y)
            glVertex2f(preview_x + preview_size, preview_y + preview_size)
            glVertex2f(preview_x, preview_y + preview_size)
            glEnd()
    
        pygame.display.flip()
        clock.tick(fps * speed_multiplier if playing else 60)
        
except KeyboardInterrupt:
    executor.shutdown(wait=False)
    pygame.quit()
    cap.release()
    sys.exit()
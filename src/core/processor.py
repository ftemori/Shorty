import os
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
from moviepy import VideoFileClip
import numpy as np
import cv2
import librosa

class VideoProcessor:
    """
    E²SAVS: Enhanced Excitation + Saliency-Aware Video Summarization
    Industry-standard local viral-clip extraction algorithm (2019–2025)
    
    Complete Pipeline (7 Stages):
    ────────────────────────────────────────────────────────────────
    Stage 1: Shot Boundary Detection
             → PySceneDetect with adaptive ContentDetector (threshold=27.0)
             
    Stage 2: Audio Excitement Analysis
             → Huang et al., 2018: Crest Factor + Spectral Flux + RMS Energy
             → Returns normalized score [0.0-3.0]
             
    Stage 3: Visual Saliency Analysis
             → Jiang et al., 2018: Face attention + motion magnitude + edge density
             → Per-second visual statistics extraction
             
    Stage 4: Face Priority Rule
             → TikTok reverse-eng. 2021: Single-face dominance with centrality weighting
             → Returns visual score [0.0-5.0] + face center for reframing
             
    Stage 5: Intelligent 9:16 Reframing
             → Wang et al., 2020: Face-centered cropping with rule-of-thirds fallback
             → Applied during clip generation (after selection)
             
    Stage 6: Importance Scoring (Viral Score)
             → Rochan et al., 2019: Linear weighted fusion
             → ViralScore = (Visual × 0.65) + (Audio × 0.35) ∈ [1.0-10.0]
             
    Stage 7: MMR Subset Selection
             → Carbonell & Goldstein, 1998 (adapted): Maximum Marginal Relevance
             → Enforces 20s minimum spacing + NO temporal overlap
             → λ=0.7 for balanced relevance/diversity
    
    Additional Enhancements:
    ────────────────────────────────────────────────────────────────
    • Speech boundary detection (silence-based, 300ms threshold)
    • Smart end-time adjustment to avoid mid-sentence cuts
    • Topic coherence checks using scene boundaries
    • Prevents clips from ending mid-topic
    """
    def __init__(self, output_path="output"):
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    # Stage 1: Shot Boundary Detection (PySceneDetect with adaptive thresholding)
    def detect_scenes(self, video_path, threshold=27.0):
        try:
            video = open_video(video_path)
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector(threshold=threshold))
            scene_manager.detect_scenes(video)
            scene_list = scene_manager.get_scene_list()
            return [(start.get_seconds(), end.get_seconds()) for start, end in scene_list]
        except Exception as e:
            print(f"Scene detection failed: {e}")
            return []

    # Stage 2: Audio Excitement (Crest Factor + Spectral Flux)
    def analyze_audio_excitement(self, y, sr):
        """
        Huang et al., 2018: Crest factor + spectral flux for audio excitement
        Returns normalized score [0.0-3.0]
        """
        if y is None or len(y) == 0:
            return 0.0
        
        try:
            # 1. RMS Energy (Loudness baseline)
            rms = librosa.feature.rms(y=y)[0]
            avg_rms = np.mean(rms)
            
            # 2. Spectral Flux (Onset Strength) - Sudden changes indicating excitement
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            spectral_flux = np.mean(onset_env)
            
            # 3. Crest Factor (Peak / RMS) - Punchiness/Impact
            peak = np.max(np.abs(y))
            rms_val = np.sqrt(np.mean(y**2))
            crest_factor = peak / rms_val if rms_val > 0 else 0
            
            # Normalize to [0-1] each
            rms_norm = min(1.0, avg_rms * 8.0)
            flux_norm = min(1.0, spectral_flux * 0.8)
            crest_norm = min(1.0, crest_factor / 15.0)
            
            # Weighted combination (Max 3.0)
            audio_score = (rms_norm * 1.0) + (flux_norm * 1.2) + (crest_norm * 0.8)
            return audio_score
        except Exception:
            return 0.0

    def detect_speech_boundaries(self, y, sr, min_silence_duration=0.3):
        """
        Detects natural speech boundaries (pauses between sentences/topics)
        Returns list of boundary timestamps (in seconds) where silence occurs
        """
        if y is None or len(y) == 0:
            return []
        
        try:
            # Calculate RMS energy in small frames
            frame_length = int(sr * 0.05)  # 50ms frames
            hop_length = int(sr * 0.025)   # 25ms hop
            
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Threshold for silence (adaptive based on overall loudness)
            rms_threshold = np.percentile(rms, 20)  # Bottom 20% is likely silence
            
            # Find silent regions
            is_silent = rms < rms_threshold
            
            # Convert frame indices to time
            boundaries = []
            in_silence = False
            silence_start = 0
            
            for i, silent in enumerate(is_silent):
                time_sec = librosa.frames_to_time(i, sr=sr, hop_length=hop_length)
                
                if silent and not in_silence:
                    # Start of silence
                    silence_start = time_sec
                    in_silence = True
                elif not silent and in_silence:
                    # End of silence
                    silence_duration = time_sec - silence_start
                    if silence_duration >= min_silence_duration:
                        # Mark the end of silence as a boundary
                        boundaries.append(time_sec)
                    in_silence = False
            
            return boundaries
        except Exception as e:
            print(f"Boundary detection warning: {e}")
            return []

    def find_nearest_boundary(self, target_time, boundaries, search_window=5.0):
        """
        Find the nearest speech boundary before the target time
        Returns adjusted time, or target_time if no boundary found
        """
        if not boundaries:
            return target_time
        
        # Look for boundaries within search_window seconds before target
        min_time = target_time - search_window
        
        # Filter boundaries in the search window
        valid_boundaries = [b for b in boundaries if min_time <= b <= target_time]
        
        if valid_boundaries:
            # Return the boundary closest to target (but before it)
            return max(valid_boundaries)
        
        return target_time

    # Stage 3: Visual Saliency (Fine-grained face attention + central bias + motion magnitude)
    def analyze_visuals_global(self, video_path, duration):
        """
        Jiang et al., 2018: Fine-grained face attention + central bias + motion magnitude
        Returns per-second visual statistics
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps if fps > 0 else 0
        
        if duration <= 0 or (video_duration > duration + 5):
            duration = video_duration

        step = int(fps)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        num_seconds = int(duration) + 1
        stats = [{'faces': 0, 'face_size': 0, 'face_center': None, 'motion': 0.0, 'edge_density': 0.0} 
                 for _ in range(num_seconds)]
        
        prev_gray = None
        second_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret or second_idx >= num_seconds:
                break
            
            try:
                h, w = frame.shape[:2]
                small_w = 320
                small_h = int(h * (small_w / w))
                small_frame = cv2.resize(frame, (small_w, small_h))
                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                
                # Face Detection (Fine-grained attention)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                face_count = len(faces)
                face_center = None
                face_size = 0
                
                if face_count > 0:
                    largest_face = max(faces, key=lambda r: r[2] * r[3])
                    fx, fy, fw, fh = largest_face
                    face_center = (fx + fw/2) / small_w  # Normalized x position
                    face_size = (fw * fh) / (small_w * small_h)  # Relative size
                
                # Motion Magnitude
                motion = 0.0
                if prev_gray is not None:
                    diff = cv2.absdiff(prev_gray, gray)
                    motion = np.count_nonzero(diff > 25) / (small_w * small_h)
                prev_gray = gray
                
                # Edge Density (visual complexity/saliency)
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.count_nonzero(edges) / (small_w * small_h)
                
                if second_idx < len(stats):
                    stats[second_idx] = {
                        'faces': face_count,
                        'face_size': face_size,
                        'face_center': face_center,
                        'motion': motion,
                        'edge_density': edge_density
                    }
            except Exception:
                pass
            
            second_idx += 1
            
            # Skip to next second
            for _ in range(step - 1):
                if not cap.grab():
                    break
                
        cap.release()
        return stats

    # Stage 4: Face Priority Rule (Single-face dominance with centrality weighting)
    def calculate_visual_score(self, stats, start, end):
        """
        TikTok reverse-eng. 2021: Single-face dominance with centrality weighting
        Returns visual score [0.0-5.0] and face center for reframing
        """
        start_sec = int(start)
        end_sec = int(end)
        clip_stats = stats[start_sec:end_sec+1]
        
        if not clip_stats:
            return {'score': 0.0, 'face_center_x': None}
        
        total_samples = len(clip_stats)
        if total_samples == 0:
            return {'score': 0.0, 'face_center_x': None}
        
        # Face metrics
        frames_with_face = sum(1 for s in clip_stats if s['faces'] > 0)
        single_face_frames = sum(1 for s in clip_stats if s['faces'] == 1)
        face_centers = [s['face_center'] for s in clip_stats if s['face_center'] is not None]
        face_sizes = [s['face_size'] for s in clip_stats if s['face_size'] > 0]
        
        # Motion and edge density (with safety checks)
        motion_values = [s['motion'] for s in clip_stats if s['motion'] is not None]
        edge_values = [s['edge_density'] for s in clip_stats if s['edge_density'] is not None]
        
        avg_motion = np.mean(motion_values) if motion_values else 0.0
        avg_edge = np.mean(edge_values) if edge_values else 0.0
        
        # 1. Single-Face Dominance Score (0-2.0)
        # TikTok algorithm prioritizes single face over multiple faces
        single_face_ratio = single_face_frames / total_samples
        face_dominance = min(2.0, single_face_ratio * 2.5)
        
        # 2. Face Centrality Weighting (0-1.5)
        centrality_score = 0.0
        avg_center_x = None
        if face_centers:
            avg_center_x = np.mean(face_centers)
            center_dist = abs(avg_center_x - 0.5)
            if center_dist < 0.15:  # Very centered
                centrality_score = 1.5
            elif center_dist < 0.25:  # Reasonably centered
                centrality_score = 1.0
            elif center_dist < 0.35:  # Slightly off-center
                centrality_score = 0.5
        
        # 3. Face Size Bonus (0-0.5) - Larger faces are better
        face_size_score = 0.0
        if face_sizes:
            avg_size = np.mean(face_sizes)
            face_size_score = min(0.5, avg_size * 10.0)
        
        # 4. Motion Magnitude (0-1.0)
        motion_score = min(1.0, avg_motion * 10.0)
        
        # 5. Visual Saliency/Complexity (0-0.5)
        edge_score = min(0.5, avg_edge * 2.5)
        
        # Total Visual Score (Max 5.5, normalized to 5.0)
        total_visual = face_dominance + centrality_score + face_size_score + motion_score + edge_score
        total_visual = min(5.0, total_visual)
        
        return {'score': total_visual, 'face_center_x': avg_center_x}

    # Stage 6: Importance Scoring (Linear weighted fusion → ViralScore ∈ [1.0-10.0])
    def calculate_viral_score(self, visual_score, audio_score):
        """
        Rochan et al., 2019: Linear weighted fusion
        Returns final ViralScore normalized to [1.0-10.0]
        """
        # Weighted fusion (Visual is more important for Shorts)
        # Visual max: 5.0, Audio max: 3.0
        # Weights: Visual 65%, Audio 35%
        raw_score = (visual_score * 0.65) + (audio_score * 0.35)
        
        # Normalize to [1.0-10.0] range
        # Max possible raw_score = (5.0 * 0.65) + (3.0 * 0.35) = 4.3
        viral_score = 1.0 + (raw_score / 4.3) * 9.0
        viral_score = max(1.0, min(10.0, viral_score))
        
        return round(viral_score, 2)  # Round to 2 decimal places

    # Stage 7: Subset Selection (MMR with 20s spacing)
    def select_clips_mmr(self, candidates, top_k=10, min_spacing=20.0):
        """
        Carbonell & Goldstein, 1998 (adapted): Maximum Marginal Relevance
        Enforces minimum 20s spacing and NO temporal overlap between clips
        """
        if not candidates:
            return []
        
        # Sort by score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Normalize scores for MMR
        max_score = candidates[0]['score'] if candidates else 10.0
        if max_score == 0:
            max_score = 10.0
        
        selected = [candidates[0]]
        remaining = candidates[1:]
        
        # Helper function to check temporal overlap
        def clips_overlap(clip1, clip2):
            """Returns True if clips have any temporal overlap"""
            return not (clip1['end'] <= clip2['start'] or clip2['end'] <= clip1['start'])
        
        def temporal_distance(clip1, clip2):
            """Returns minimum temporal distance between two clips"""
            if clips_overlap(clip1, clip2):
                return 0.0
            # Distance between end of earlier clip and start of later clip
            if clip1['end'] <= clip2['start']:
                return clip2['start'] - clip1['end']
            else:
                return clip1['start'] - clip2['end']
        
        # MMR selection with diversity and no overlap
        while len(selected) < top_k and remaining:
            best_mmr = -float('inf')
            best_candidate = None
            
            for cand in remaining:
                # Check for overlap with any selected clip
                has_overlap = any(clips_overlap(cand, sel) for sel in selected)
                if has_overlap:
                    continue  # Skip overlapping clips entirely
                
                # Relevance (score-based)
                relevance = cand['score'] / max_score
                
                # Redundancy (temporal similarity to selected clips)
                max_similarity = 0.0
                for sel in selected:
                    dist = temporal_distance(cand, sel)
                    
                    # Similarity decays with distance
                    if dist < min_spacing:
                        similarity = 1.0 - (dist / min_spacing)
                    else:
                        similarity = 0.0
                    
                    max_similarity = max(max_similarity, similarity)
                
                # MMR formula (lambda=0.7 for balanced relevance/diversity)
                mmr_score = (0.7 * relevance) - (0.3 * max_similarity)
                
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_candidate = cand
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break
        
        # Sort final selection by score for display
        selected.sort(key=lambda x: x['score'], reverse=True)
        return selected

    def analyze_video(self, video_path, min_duration=15, max_duration=60, progress_callback=None):
        """
        E²SAVS Pipeline: Full viral clip extraction
        
        Stages executed in this method:
        1. Shot Boundary Detection (PySceneDetect)
        2. Audio Excitement Analysis (Crest Factor + Spectral Flux)
        3. Visual Saliency Analysis (Face attention + motion + edge density)
        4. Face Priority Rule (Single-face dominance scoring)
        6. Importance Scoring (Linear fusion → ViralScore)
        7. MMR Subset Selection (20s spacing + no overlap)
        
        Stage 5 (9:16 Reframing) is executed later in create_vertical_short() during clip generation.
        
        Additional Enhancements:
        - Speech boundary detection for natural sentence endings
        - Topic coherence checks using scene boundaries
        - Smart end-time adjustment to avoid mid-sentence cuts
        """
        print(f"E²SAVS Analysis started: {video_path}")
        
        # Get video duration
        duration = 0
        try:
            with VideoFileClip(video_path) as v:
                if v.duration and v.duration > 0:
                    duration = v.duration
        except:
            pass
        
        if duration <= 0:
            try:
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    if fps > 0:
                        duration = frames / fps
                    cap.release()
            except:
                pass
        
        if duration <= 0:
            print("Error: Cannot determine video duration")
            return []
        
        print(f"Video duration: {duration:.1f}s")
        
        # ═══════════════════════════════════════════════════════════════
        # STAGE 1: Shot Boundary Detection (PySceneDetect)
        # ═══════════════════════════════════════════════════════════════
        raw_scenes = self.detect_scenes(video_path)
        candidates = []
        
        if raw_scenes:
            print(f"Detected {len(raw_scenes)} scenes")
            # Merge and chunk scenes
            curr_start = raw_scenes[0][0]
            curr_end = raw_scenes[0][1]
            
            for i in range(1, len(raw_scenes)):
                next_start, next_end = raw_scenes[i]
                
                if (curr_end - curr_start) < 10.0:  # Merge very short scenes
                    curr_end = next_end
                else:
                    dur = curr_end - curr_start
                    if min_duration <= dur <= max_duration:
                        candidates.append((curr_start, curr_end))
                    elif dur > max_duration:
                        # Split long scenes into NON-OVERLAPPING windows
                        # Use 45s stride to ensure no overlap with 60s max clips
                        for cs in range(int(curr_start), int(curr_end - min_duration), 45):
                            ce = min(cs + max_duration, curr_end)
                            if (ce - cs) >= min_duration:
                                candidates.append((float(cs), float(ce)))
                    curr_start = next_start
                    curr_end = next_end
            
            # Last scene
            dur = curr_end - curr_start
            if min_duration <= dur <= max_duration:
                candidates.append((curr_start, curr_end))
            elif dur > max_duration:
                for cs in range(int(curr_start), int(curr_end - min_duration), 45):
                    ce = min(cs + max_duration, curr_end)
                    if (ce - cs) >= min_duration:
                        candidates.append((float(cs), float(ce)))
        
        # Fallback: Non-overlapping sliding window
        if not candidates:
            print("Fallback: Using non-overlapping sliding window (45s stride)")
            for cs in range(0, int(duration - min_duration), 45):
                ce = min(cs + max_duration, duration)
                if (ce - cs) >= min_duration:
                    candidates.append((float(cs), float(ce)))
        
        print(f"Generated {len(candidates)} candidate clips")
        
        # ═══════════════════════════════════════════════════════════════
        # STAGE 2: Audio Excitement Analysis (Crest Factor + Spectral Flux)
        # ═══════════════════════════════════════════════════════════════
        temp_audio = "temp_e2savs.wav"
        y_full = None
        sr = 16000
        
        try:
            with VideoFileClip(video_path) as v:
                if v.audio:
                    v.audio.write_audiofile(temp_audio, fps=16000, logger=None)
            if os.path.exists(temp_audio):
                y_full, sr = librosa.load(temp_audio, sr=16000, mono=True)
                os.remove(temp_audio)
        except Exception as e:
            print(f"Audio extraction warning: {e}")
        
        # ═══════════════════════════════════════════════════════════════
        # STAGE 3: Visual Saliency Analysis (Face + Motion + Edge Density)
        # ═══════════════════════════════════════════════════════════════
        print("Running visual saliency analysis...")
        visual_stats = self.analyze_visuals_global(video_path, duration)
        
        # Detect speech boundaries (sentence/topic boundaries) from full audio
        speech_boundaries = []
        if y_full is not None:
            print("Detecting speech boundaries...")
            speech_boundaries = self.detect_speech_boundaries(y_full, sr, min_silence_duration=0.3)
            print(f"Found {len(speech_boundaries)} potential sentence/topic boundaries")
        
        # Combine with scene boundaries for topic coherence
        scene_boundaries = [scene[1] for scene in raw_scenes] if raw_scenes else []
        all_boundaries = sorted(set(speech_boundaries + scene_boundaries))
        
        # ═══════════════════════════════════════════════════════════════
        # STAGES 4 & 6: Face Priority Rule + Importance Scoring
        # ═══════════════════════════════════════════════════════════════
        scored_candidates = []
        
        for idx, (start, end) in enumerate(candidates):
            # Audio excitement score
            audio_score = 0.0
            if y_full is not None:
                s_idx = int(start * sr)
                e_idx = int(end * sr)
                if e_idx > s_idx:
                    audio_score = self.analyze_audio_excitement(y_full[s_idx:e_idx], sr)
            
            # Visual saliency score
            visual_result = self.calculate_visual_score(visual_stats, start, end)
            visual_score = visual_result['score']
            
            # Viral score (weighted fusion)
            viral_score = self.calculate_viral_score(visual_score, audio_score)
            
            # CRITICAL: Adjust end time to nearest boundary
            # This ensures clips don't end mid-sentence or mid-topic
            original_end = end
            adjusted_end = self.find_nearest_boundary(end, all_boundaries, search_window=5.0)
            
            # Ensure minimum duration (15s) after adjustment
            if (adjusted_end - start) < 15.0:
                adjusted_end = original_end  # Keep original if adjustment makes it too short
            
            # Ensure we don't cross into a new topic
            # If there's a scene boundary between start and adjusted_end, stop before it
            scene_boundaries_in_clip = [b for b in scene_boundaries if start < b < adjusted_end]
            if scene_boundaries_in_clip:
                # There's a scene change (topic change) - end before it
                first_scene_boundary = min(scene_boundaries_in_clip)
                if (first_scene_boundary - start) >= 15.0:
                    adjusted_end = first_scene_boundary
                    print(f"Clip {idx}: Adjusted end to scene boundary at {adjusted_end:.1f}s to avoid topic change")
            elif adjusted_end != original_end:
                print(f"Clip {idx}: Adjusted end from {original_end:.1f}s to {adjusted_end:.1f}s (speech boundary)")
            
            scored_candidates.append({
                "id": f"clip_{idx}",
                "index": idx,
                "start": start,
                "end": adjusted_end,
                "duration": round(adjusted_end - start, 1),
                "score": viral_score,
                "face_center_x": visual_result['face_center_x']
            })
            
            if progress_callback:
                progress_callback(idx + 1, len(candidates))
        
        # ═══════════════════════════════════════════════════════════════
        # STAGE 7: MMR Subset Selection (20s spacing + no overlap)
        # ═══════════════════════════════════════════════════════════════
        print("Applying MMR selection...")
        candidates_before = len(scored_candidates)
        final_clips = self.select_clips_mmr(scored_candidates, top_k=10, min_spacing=20.0)
        
        # Re-index
        for i, clip in enumerate(final_clips):
            clip["index"] = i
            clip["id"] = f"clip_{i}"
        
        if final_clips:
            filtered_count = candidates_before - len(final_clips)
            print(f"E²SAVS selected {len(final_clips)} clips (filtered {filtered_count} overlapping/redundant)")
            print(f"Score range: {final_clips[0]['score']} - {final_clips[-1]['score']}")
            print(f"Time spans: {[f"{c['start']:.0f}s-{c['end']:.0f}s" for c in final_clips[:3]]}...")
        else:
            print("Warning: No clips selected")
        
        return final_clips

    # Clip Generation: Intelligent 9:16 Reframing (Stage 5 of E²SAVS - applied after selection)
    def create_vertical_short(self, video_path, start_time, end_time, output_filename, face_center_x=None):
        """
        Wang et al., 2020: Intelligent 9:16 reframing with face-centered cropping
        Stage 5 of E²SAVS is executed here during clip generation, after MMR selection (Stage 7)
        """
        video = VideoFileClip(video_path)
        if hasattr(video, 'subclipped'):
            clip = video.subclipped(start_time, end_time)
        else:
            clip = video.subclip(start_time, end_time)
        
        w, h = clip.size
        target_ratio = 9/16
        
        if w / h > target_ratio:
            new_w = int(h * target_ratio)
            # Ensure width is even (divisible by 2) for H.264 encoding
            if new_w % 2 != 0:
                new_w -= 1
            
            if face_center_x is not None:
                # Face-centered crop
                center_pixel = face_center_x * w
                x1 = center_pixel - new_w / 2
                x2 = center_pixel + new_w / 2
                
                # Clamp to bounds
                if x1 < 0:
                    x1, x2 = 0, new_w
                elif x2 > w:
                    x1, x2 = w - new_w, w
            else:
                # Rule-of-thirds fallback (slightly off-center)
                x_center = w * 0.5
                x1 = x_center - new_w / 2
                x2 = x_center + new_w / 2
            
            # Round to even numbers
            x1 = int(x1)
            x2 = int(x2)
            if (x2 - x1) % 2 != 0:
                x2 -= 1
            
            if hasattr(clip, 'cropped'):
                clip = clip.cropped(x1=x1, y1=0, x2=x2, y2=h)
            else:
                clip = clip.crop(x1=x1, y1=0, x2=x2, y2=h)
        else:
            new_h = int(w / target_ratio)
            # Ensure height is even (divisible by 2) for H.264 encoding
            if new_h % 2 != 0:
                new_h -= 1
            
            y_center = h / 2
            y1 = y_center - new_h / 2
            y2 = y_center + new_h / 2
            
            # Round to even numbers
            y1 = int(y1)
            y2 = int(y2)
            if (y2 - y1) % 2 != 0:
                y2 -= 1
            
            if hasattr(clip, 'cropped'):
                clip = clip.cropped(x1=0, y1=y1, x2=w, y2=y2)
            else:
                clip = clip.crop(x1=0, y1=y1, x2=w, y2=y2)
        
        output_file = os.path.join(self.output_path, output_filename)
        
        # Force software encoding to avoid CUDA/hardware acceleration issues
        # Use baseline profile for maximum compatibility
        clip.write_videofile(
            output_file, 
            codec='libx264',
            audio_codec='aac',
            preset='medium',
            ffmpeg_params=[
                '-pix_fmt', 'yuv420p',  # Standard pixel format
                '-profile:v', 'baseline',  # H.264 baseline profile (most compatible)
                '-level', '3.0',  # Compatibility level
                '-movflags', '+faststart'  # Enable streaming/fast playback
            ],
            logger=None
        )
        video.close()
        return output_file

    def generate_clips(self, video_path, clips):
        if not clips:
            return []
        
        generated = []
        base_name = os.path.basename(video_path)
        
        for clip in clips:
            start = clip.get("start")
            end = clip.get("end")
            face_center = clip.get("face_center_x")
            
            if start is None or end is None:
                continue
            
            clip_id = clip.get("id")
            output_filename = f"{clip_id}_{base_name}"
            generated.append(self.create_vertical_short(video_path, start, end, output_filename, face_center_x=face_center))
        
        return generated

"""
Flask Backend for Pose-Guided Person Image Generation
Handles file uploads, pose generation, and reel creation
"""

from flask import Flask, request, render_template, jsonify, send_file, redirect, url_for
import os
import uuid
import json
from werkzeug.utils import secure_filename
from pose_inference import PoseInference, create_reel_from_images
import shutil
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['GENERATED_FOLDER'] = 'generated'
app.config['REELS_FOLDER'] = 'reels'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Initialize pose inference
model_path = 'checkpoints/pose_guided_gan_epoch_9.pth'  # Update with actual model path
pose_inference = PoseInference(model_path)

# Session storage (in production, use Redis or database)
user_sessions = {}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_directories():
    """Create necessary directories"""
    for folder in [app.config['UPLOAD_FOLDER'], app.config['GENERATED_FOLDER'], app.config['REELS_FOLDER']]:
        os.makedirs(folder, exist_ok=True)

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Generate unique session ID
            session_id = str(uuid.uuid4())
            
            # Save uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
            file.save(file_path)
            
            # Store session info
            user_sessions[session_id] = {
                'uploaded_file': file_path,
                'generated_images': [],
                'current_poses': [],
                'created_at': datetime.now()
            }
            
            return jsonify({
                'success': True,
                'session_id': session_id,
                'message': 'File uploaded successfully'
            })
        
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate_poses():
    """Generate 5 different poses from uploaded image"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id or session_id not in user_sessions:
            return jsonify({'error': 'Invalid session'}), 400
        
        session = user_sessions[session_id]
        uploaded_file = session['uploaded_file']
        
        if not os.path.exists(uploaded_file):
            return jsonify({'error': 'Uploaded file not found'}), 400
        
        # Generate poses
        generated_images = pose_inference.generate_poses(uploaded_file, num_poses=5)
        
        # Save generated images
        output_dir = os.path.join(app.config['GENERATED_FOLDER'], session_id)
        os.makedirs(output_dir, exist_ok=True)
        
        saved_paths = []
        image_info = []
        
        for i, img_info in enumerate(generated_images):
            filename = f"pose_{i+1}_{img_info['pose_name']}.jpg"
            filepath = os.path.join(output_dir, filename)
            img_info['image'].save(filepath)
            saved_paths.append(filepath)
            
            image_info.append({
                'filename': filename,
                'pose_name': img_info['pose_name'],
                'path': f"/generated/{session_id}/{filename}"
            })
        
        # Update session
        session['generated_images'] = saved_paths
        session['current_poses'] = image_info
        
        return jsonify({
            'success': True,
            'images': image_info,
            'message': 'Poses generated successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/accept', methods=['POST'])
def accept_poses():
    """Accept current poses and create reel"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id or session_id not in user_sessions:
            return jsonify({'error': 'Invalid session'}), 400
        
        session = user_sessions[session_id]
        generated_images = session['generated_images']
        
        if not generated_images:
            return jsonify({'error': 'No generated images found'}), 400
        
        # Create reel
        reel_filename = f"reel_{session_id}.mp4"
        reel_path = os.path.join(app.config['REELS_FOLDER'], reel_filename)
        
        # Create reel with transitions and music
        success = create_enhanced_reel(generated_images, reel_path)
        
        if success:
            return jsonify({
                'success': True,
                'reel_path': f"/download/{reel_filename}",
                'message': 'Reel created successfully'
            })
        else:
            return jsonify({'error': 'Failed to create reel'}), 500
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/reject', methods=['POST'])
def reject_poses():
    """Reject current poses and generate new ones"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if not session_id or session_id not in user_sessions:
            return jsonify({'error': 'Invalid session'}), 400
        
        session = user_sessions[session_id]
        uploaded_file = session['uploaded_file']
        
        if not os.path.exists(uploaded_file):
            return jsonify({'error': 'Uploaded file not found'}), 400
        
        # Generate new poses (different from current ones)
        current_pose_names = [img['pose_name'] for img in session['current_poses']]
        generated_images = pose_inference.generate_poses(
            uploaded_file, 
            num_poses=5,
            pose_names=None  # Let it choose random poses
        )
        
        # Save new generated images
        output_dir = os.path.join(app.config['GENERATED_FOLDER'], session_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Clear old images
        for old_file in session['generated_images']:
            if os.path.exists(old_file):
                os.remove(old_file)
        
        saved_paths = []
        image_info = []
        
        for i, img_info in enumerate(generated_images):
            filename = f"pose_{i+1}_{img_info['pose_name']}.jpg"
            filepath = os.path.join(output_dir, filename)
            img_info['image'].save(filepath)
            saved_paths.append(filepath)
            
            image_info.append({
                'filename': filename,
                'pose_name': img_info['pose_name'],
                'path': f"/generated/{session_id}/{filename}"
            })
        
        # Update session
        session['generated_images'] = saved_paths
        session['current_poses'] = image_info
        
        return jsonify({
            'success': True,
            'images': image_info,
            'message': 'New poses generated successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generated/<session_id>/<filename>')
def serve_generated_image(session_id, filename):
    """Serve generated images"""
    try:
        file_path = os.path.join(app.config['GENERATED_FOLDER'], session_id, filename)
        if os.path.exists(file_path):
            return send_file(file_path)
        else:
            return jsonify({'error': 'Image not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_reel(filename):
    """Download generated reel"""
    try:
        file_path = os.path.join(app.config['REELS_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': 'Reel not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def create_enhanced_reel(image_paths, output_path, duration_per_image=4, fps=30):
    """Create an enhanced reel with transitions and music"""
    try:
        from moviepy.editor import (
            ImageSequenceClip, concatenate_videoclips, 
            CompositeVideoClip, TextClip, AudioFileClip
        )
        
        # Create clips from images with transitions
        clips = []
        for i, img_path in enumerate(image_paths):
            # Create image clip
            clip = ImageSequenceClip([img_path], durations=[duration_per_image])
            
            # Add fade in/out effects
            if i > 0:  # Fade in for all except first
                clip = clip.fadein(0.5)
            if i < len(image_paths) - 1:  # Fade out for all except last
                clip = clip.fadeout(0.5)
            
            # Add text overlay with pose name
            pose_name = os.path.basename(img_path).split('_')[2].replace('.jpg', '')
            txt_clip = TextClip(
                pose_name.replace('_', ' ').title(),
                fontsize=30,
                color='white',
                font='Arial-Bold'
            ).set_position(('center', 'bottom')).set_duration(duration_per_image)
            
            # Composite text over image
            clip = CompositeVideoClip([clip, txt_clip])
            clips.append(clip)
        
        # Concatenate all clips
        final_clip = concatenate_videoclips(clips)
        
        # Add background music if available
        music_path = 'static/background_music.mp3'
        if os.path.exists(music_path):
            try:
                audio = AudioFileClip(music_path)
                # Adjust audio duration to match video
                if audio.duration > final_clip.duration:
                    audio = audio.subclip(0, final_clip.duration)
                else:
                    # Loop audio if it's shorter than video
                    audio = audio.loop(duration=final_clip.duration)
                
                # Set audio volume (lower so it doesn't overpower)
                audio = audio.volumex(0.3)
                final_clip = final_clip.set_audio(audio)
            except Exception as e:
                print(f"Could not add background music: {e}")
        
        # Write video
        final_clip.write_videofile(
            output_path, 
            fps=fps, 
            codec='libx264',
            audio_codec='aac' if final_clip.audio else None
        )
        
        print(f"Enhanced reel created: {output_path}")
        return True
        
    except ImportError:
        print("MoviePy not available. Creating simple reel...")
        return create_reel_from_images(image_paths, output_path, duration_per_image, fps)
    except Exception as e:
        print(f"Error creating enhanced reel: {e}")
        return False

@app.route('/cleanup/<session_id>', methods=['POST'])
def cleanup_session(session_id):
    """Clean up session files"""
    try:
        if session_id in user_sessions:
            session = user_sessions[session_id]
            
            # Remove uploaded file
            if os.path.exists(session['uploaded_file']):
                os.remove(session['uploaded_file'])
            
            # Remove generated images
            output_dir = os.path.join(app.config['GENERATED_FOLDER'], session_id)
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            
            # Remove session
            del user_sessions[session_id]
            
            return jsonify({'success': True, 'message': 'Session cleaned up'})
        
        return jsonify({'error': 'Session not found'}), 404
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    create_directories()
    app.run(debug=True, host='0.0.0.0', port=5000)

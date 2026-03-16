#!/usr/bin/env python3
"""
Test AV1 compression support in RendererM
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'modules'))

from RendererM import _detect_available_encoders, _apply_final_compression

def test_encoder_detection():
    """Test that we can detect available encoders"""
    print("🔍 Testing encoder detection...")
    encoders = _detect_available_encoders()
    
    print(f"Available AV1 encoders: {encoders['available_av1']}")
    print(f"Available H.264 encoders: {encoders['available_h264']}")
    print(f"Preferred AV1: {encoders['preferred_av1']}")
    print(f"Preferred H.264: {encoders['preferred_h264']}")
    
    if encoders['preferred_av1']:
        print("✅ AV1 encoder detected and available")
        return True
    else:
        print("⚠️  No AV1 encoder found, will use H.264")
        return False

def test_compression():
    """Test compression on the problematic clip"""
    input_file = "gaming_pipeline_0314_0249/video_clips/clip_1.mp4"
    output_file = "test_compressed_clip.mp4"
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return False
    
    print(f"🗜️  Testing compression on {input_file}")
    
    # Test with AV1 if available
    encoders = _detect_available_encoders()
    encoder_config = {
        'use_av1': bool(encoders['preferred_av1']),
        'final_compression': True
    }
    
    try:
        _apply_final_compression(input_file, output_file, encoder_config)
        
        if os.path.exists(output_file):
            original_size = os.path.getsize(input_file)
            compressed_size = os.path.getsize(output_file)
            ratio = (1 - compressed_size / original_size) * 100
            
            print(f"✅ Compression successful!")
            print(f"   Original: {original_size/1024/1024:.1f}MB")
            print(f"   Compressed: {compressed_size/1024/1024:.1f}MB")
            print(f"   Reduction: {ratio:.1f}%")
            
            # Clean up test file
            os.remove(output_file)
            return True
        else:
            print("❌ Output file not created")
            return False
            
    except Exception as e:
        print(f"❌ Compression failed: {e}")
        return False

if __name__ == "__main__":
    print("🎬 Testing AV1 compression support in ClipperAI")
    print("=" * 50)
    
    av1_available = test_encoder_detection()
    print()
    
    if test_compression():
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    
    print()
    print("Usage in your projects:")
    print("1. The system will automatically use AV1 if available")
    print("2. You can force H.264 with --no-av1 flag")
    print("3. You can skip compression with --no-compression flag")

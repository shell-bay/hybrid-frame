#!/bin/bash

# HybridFrame Video Render Script
# Requires: Node.js >= 22, FFmpeg

echo "🎬 HybridFrame Video Render"
echo "=========================="
echo ""

# Check prerequisites
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required. Install it from https://nodejs.org/"
    exit 1
fi

if ! command -v ffmpeg &> /dev/null; then
    echo "❌ FFmpeg is required. Install it with: brew install ffmpeg"
    exit 1
fi

# Check if hyperframes is installed
if ! command -v npx &> /dev/null; then
    echo "❌ npx is required. It comes with Node.js."
    exit 1
fi

echo "✅ Prerequisites checked"
echo ""

# Initialize HyperFrames project
echo "📦 Initializing HyperFrames..."
cd video
npx hyperframes init hybridframe-video 2>/dev/null || true
cd hybridframe-video

# Copy composition
cp ../composition.html .

# Preview in browser
echo "👀 Opening preview in browser..."
echo "   (Press Ctrl+C when done previewing)"
echo ""
npx hyperframes preview

# Render video
echo ""
echo "🎥 Rendering video..."
npx hyperframes render --output ../../hybridframe-demo.mp4

echo ""
echo "✅ Video rendered: hybridframe-demo.mp4"
echo ""
echo "Upload to GitHub:"
echo "  1. Go to your README.md"
echo "  2. Add: ![Demo](hybridframe-demo.mp4)"
echo "  3. Commit and push"

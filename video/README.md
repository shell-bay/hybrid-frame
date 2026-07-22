# HybridFrame Video Demo

This directory contains a HyperFrames composition for creating a promotional video for HybridFrame.

## Prerequisites

1. **Node.js** >= 22 - [Install](https://nodejs.org/)
2. **FFmpeg** - Install with `brew install ffmpeg`

## Quick Start

### Option 1: Using the render script

```bash
chmod +x render.sh
./render.sh
```

### Option 2: Manual steps

```bash
# Install HyperFrames
npm install -g hyperframes

# Initialize project
npx hyperframes init my-video
cd my-video

# Copy composition
cp ../composition.html .

# Preview in browser
npx hyperframes preview

# Render to MP4
npx hyperframes render --output hybridframe-demo.mp4
```

## Adding to README

Once rendered, add the video to your README.md:

```markdown
## Demo

https://github.com/user-attachments/assets/hybridframe-demo.mp4
```

Or as a GIF:

```markdown
## Demo

![HybridFrame Demo](hybridframe-demo.gif)
```

To convert MP4 to GIF:

```bash
ffmpeg -i hybridframe-demo.mp4 -vf "fps=10,scale=800:-1" hybridframe-demo.gif
```

## Composition Structure

The video consists of 7 slides:

1. **Title** - HybridFrame branding
2. **Problem** - Why Pandas alone isn't enough
3. **Solution** - How HybridFrame solves it
4. **Code Example** - Simple API demonstration
5. **Performance** - Benchmark comparison
6. **Features** - Key capabilities
7. **CTA** - Installation and GitHub link

## Customization

Edit `composition.html` to:
- Change colors (modify CSS variables)
- Update text content
- Adjust timing (change `data-duration` attributes)
- Add your own branding

## License

MIT License - Same as HybridFrame

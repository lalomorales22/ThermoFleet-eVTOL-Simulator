# 📚 ThermoFleet Documentation Hub

Welcome to the ThermoFleet eVTOL Simulator documentation! All project documentation is now available in a beautiful, searchable web interface.

---

## 🚀 Quick Start

### ⚠️ Important: Run a Local Server

**You MUST serve the documentation through HTTP**, not open it directly as a file. This is due to browser CORS security restrictions that prevent loading markdown files from `file://` URLs.

### **Easiest Method: Use the Serve Script**

```bash
# From project root
./docs/serve.sh

# Then open your browser to:
# http://localhost:8000
```

### **Alternative Methods**

#### Using Python (Recommended)
```bash
cd docs
python3 -m http.server 8000
# Open: http://localhost:8000
```

#### Using Node.js
```bash
cd docs
npx http-server -p 8000
# Open: http://localhost:8000
```

#### Using PHP
```bash
cd docs
php -S localhost:8000
# Open: http://localhost:8000
```

### ❌ **What Won't Work**

```bash
# ❌ DON'T do this - will get CORS errors
open docs/index.html
```

**Why?** Browsers block JavaScript from loading local files when opened via `file://` protocol for security reasons.

---

## ✨ Features

### 🎨 Dark Theme
Professional dark theme optimized for long reading sessions

### 🔍 Search
Instant search across all documentation files

### 📱 Responsive
Works perfectly on desktop, tablet, and mobile

### 🎯 Organized
Documentation categorized by:
- **Getting Started** - Setup guides and project overview
- **Core Features** - Thermodynamic computing, data flow, Omniverse
- **Monitoring & Visualization** - TensorBoard, WandB, Dashboard
- **Troubleshooting & Fixes** - All recent fixes and solutions
- **Development** - Contributing, CI/CD, reviews
- **Project History** - Session summaries and phase reviews

### 💻 Syntax Highlighting
Beautiful code syntax highlighting for Python, Bash, and more

### 🔗 Smart Navigation
- Sidebar navigation with icons
- Active document highlighting
- Smooth scrolling
- Scroll-to-top button

---

## 📖 Available Documentation

### Getting Started
- 📖 Getting Started Guide
- 🏠 Project Overview (README)
- ⚙️ Environment Setup

### Core Features
- 🔥 Thermodynamic Computing Usage
- 📊 Data Flow Architecture
- 🌐 Omniverse Integration Setup

### Monitoring & Visualization
- 📈 TensorBoard Guide
- ☁️ WandB Setup & Configuration
- 🎨 UI Guide
- 📊 Dashboard Fixes

### Troubleshooting & Fixes
- ✅ All Fixes Complete (Summary)
- 🔧 Pickle Error Fix (Training)
- ☁️ WandB Error Fix
- 🔥 Thermodynamic NaN Fix

### Development
- 🤝 Contributing Guide
- 🚀 CI/CD Guide
- 📱 App Review

### Project History
- 📅 Session Summaries
- 🎯 Phase Summaries
- 🤖 AI Reviews

---

## 🎯 How to Use

### 1. **Browse by Category**
Click on any section in the sidebar to explore documentation

### 2. **Search**
Use the search bar at the top to find specific topics instantly

### 3. **Navigate**
Click on document titles in the sidebar to switch between docs

### 4. **Quick Return**
Use the "↑" button in the bottom-right to quickly scroll to top

---

## 🛠️ Technical Details

### Built With
- **Marked.js** - Markdown parser
- **Highlight.js** - Syntax highlighting
- **Pure CSS** - No frameworks, lightweight and fast
- **Vanilla JS** - No dependencies beyond markdown parsing

### Browser Support
- ✅ Chrome/Edge 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Mobile browsers

---

## 📝 Adding New Documentation

To add new documentation:

1. **Create your .md file** in the `docs/` folder
2. **Edit `index.html`** and add your doc to the `docs` object:

```javascript
const docs = {
    'Your Category': [
        { file: 'YOUR_FILE.md', title: '🎯 Your Title' },
    ],
    // ...
};
```

3. **Refresh** the documentation site - your new doc will appear!

---

## 🎨 Customization

### Change Theme Colors
Edit the CSS variables in `index.html`:

```css
:root {
    --bg-primary: #0d1117;      /* Main background */
    --bg-secondary: #161b22;    /* Sidebar/cards */
    --accent-primary: #58a6ff;  /* Links/highlights */
    /* ... */
}
```

### Add New Categories
Simply add to the `docs` object in the JavaScript section

### Modify Layout
Adjust the `--sidebar-width` CSS variable

---

## 🚀 Deployment

### GitHub Pages
```bash
# Commit the docs folder
git add docs/
git commit -m "Add documentation site"
git push origin main

# Enable GitHub Pages pointing to /docs folder
# Documentation will be available at:
# https://your-username.github.io/ThermoFleet-eVTOL-Simulator/docs/
```

### Static Hosting
Upload the entire `docs/` folder to any static hosting service:
- Netlify
- Vercel
- AWS S3
- Any web server

---

## 💡 Tips

### Keyboard Shortcuts (Coming Soon)
- `Ctrl/Cmd + K` - Focus search
- `Ctrl/Cmd + B` - Toggle sidebar
- `Esc` - Clear search

### Mobile Navigation
On mobile devices, swipe from the left edge to open the sidebar

### Offline Use
All documentation works offline - just open `index.html` locally!

---

## 🤝 Contributing

If you improve the documentation site:
1. Keep the dark theme consistent
2. Ensure mobile responsiveness
3. Test across different browsers
4. Update this README if you add features

---

## 📞 Questions?

- Check the main [README.md](../README.md)
- See [CONTRIBUTING.md](CONTRIBUTING.md)
- Review [GETTING_STARTED.md](../GETTING_STARTED.md)

---

**Happy exploring!** 📚✨


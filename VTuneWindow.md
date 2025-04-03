# How to Install Intel VTune Profiler on Windows

Intel VTune is a powerful performance analysis tool. This guide explains how to install and run it on Windows.

---

## ✅ Option 1: Install via oneAPI Base Toolkit (Recommended)

### Step 1: Download

Go to: https://www.intel.com/vtune

- Click **Download**
- Choose **Windows**
- Select either:
  - **Base Toolkit** (includes VTune and other tools), or
  - **Standalone VTune Profiler**

---

### Step 2: Install

1. Run the installer `.exe`
2. Choose "Customize" if you want only VTune
3. Accept the license and finish installation

VTune is typically installed to:
```
C:\Program Files (x86)\Intel\oneAPI\vtune\latest\
```

---

### Step 3: Verify Installation

- `vtune-gui.exe` = GUI
- `vtune.exe` = Command-line tool

---

## 🚀 How to Run VTune

### GUI:

- Launch from Start Menu:  
  **Start → Intel oneAPI VTune Profiler**

Or run manually:
```
C:\Program Files (x86)\Intel\oneAPI\vtune\latest\bin64\vtune-gui.exe
```


---

### Command-Line:

Open **"x64 Native Tools Command Prompt for VS"** and run:

```
vtune -collect hotspots -result-dir r001 ./your_program.exe
```

Replace `./your_program.exe` with your application.

---

## 📌 Optional: Add VTune to PATH

To run `vtune` from any terminal:

1. Open System Properties → Environment Variables
2. Add this path:

```
C:\Program Files (x86)\Intel\oneAPI\vtune\latest\bin64
```

---

## 🔐 Optional: Enable Full Hardware Profiling

For cache/memory analysis, VTune may prompt to install a sampling driver. Allow it when prompted.

- Requires administrator rights
- Enables memory-bound, cache miss, and bandwidth metrics

---

## ✅ System Requirements

- Windows 10 or later
- Admin access
- Visual Studio (for builds/symbols)


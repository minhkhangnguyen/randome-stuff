# Workflow Between You and Me

## How We Work Together

### 1. I Do All the Work
- I develop and fix the code
- I push updates to GitHub
- I maintain the project

### 2. You Only Need to Pull
- You run `git pull` to get the latest version
- You run `run.bat` to start the translator

---

## Current Files in the Project

| File                    | Purpose                                      |
|-------------------------|----------------------------------------------|
| `video_translator.py`   | Main translator (audio + translation)        |
| `run.bat`               | One-click launcher (includes auto `git pull`)|
| `install_models.bat`    | Downloads Chinese/Japanese → Vietnamese models |
| `requirements.txt`      | Python dependencies                          |
| `WORKFLOW.md`           | This file                                    |

---

## How to Use (Daily)

### Step 1: Update the project
```cmd
cd /d D:\randome-stuff
git pull
```

### Step 2: Run the translator
Just double-click **`run.bat`**

`run.bat` will automatically:
1. Run `git pull` (check for updates)
2. Start the translator

---

## First Time Setup (Only Once)

1. Clone the repo:
   ```cmd
   cd /d D:\
   git clone https://github.com/minhkhangnguyen/randome-stuff.git
   cd randome-stuff
   ```

2. Install Python packages:
   ```cmd
   "D:\AI generated\local_runtime\Python\Python312\python.exe" -m pip install -r requirements.txt
   ```

3. Download translation models:
   Double-click `install_models.bat`

4. Run the translator:
   Double-click `run.bat`

---

## Notes

- Everything stays **local** after first setup
- Your audio/transcript never leaves your PC
- The app works even when speakers are muted (uses system audio)

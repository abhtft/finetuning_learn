# WSL2 & Ubuntu Terminal Practical Guidelines

A practical guide and reference manual for working productively in **Windows Subsystem for Linux (WSL2)** and **Ubuntu**. This guide covers core mental models, daily terminal commands, performance optimization, GPU/CUDA management, file system rules, and best practices for development and machine learning.

---

## 1. The Core Mental Model (WSL2 Architecture)

WSL2 runs a **real Linux kernel** inside a lightweight hypervisor directly alongside Windows. Understanding how they communicate is key to avoiding common pitfalls.

```
┌────────────────────────────────────────────────────────┐
│                     WINDOWS HOST                       │
│  - Windows Terminal / VS Code / Cursor / IDE           │
│  - NVIDIA Host Driver (DirectX / CUDA D3D12 interface) │
│  - Storage: C:\, D:\ (NTFS)                            │
└───────────────────────────▲────────────────────────────┘
                            │ Plan 9 / Virtual Network (~9P Protocol)
┌───────────────────────────▼────────────────────────────┐
│                    WSL2 (UBUNTU)                       │
│  - Linux Kernel + Native ext4 Virtual Disk (VHDX)      │
│  - High-performance storage: /home/abhishek/           │
│  - CUDA Toolkit / PyTorch / Docker / Python Venvs      │
└────────────────────────────────────────────────────────┘
```

### The #1 Golden Rule of WSL Performance
> [!IMPORTANT]
> **Always store your code and virtual environments inside the Linux filesystem (`/home/username/...` or `~`), NEVER inside Windows mounts (`/mnt/c/...`).**
>
> * **Linux filesystem (`~/projects/`)**: Native Linux `ext4` filesystem. I/O speed is **10x – 20x faster**, file watchers work properly, and git operations are instantaneous.
> * **Windows mount (`/mnt/c/Users/...`)**: Crosses the 9P network boundary. Slow I/O, causes severe performance drops during `pip install`, Git scans, model training, and dataset loading.

---

## 2. Cross-OS Interoperability Superpowers

WSL and Windows can interact seamlessly in both directions:

### A. Accessing Files Across OS Boundaries

| Direction | Command / Path | What It Does |
| :--- | :--- | :--- |
| **Linux $\rightarrow$ Windows Explorer** | `explorer.exe .` | Opens the current Linux directory in Windows File Explorer |
| **Linux $\rightarrow$ Windows Path** | `/mnt/c/Users/YourName/Desktop/` | Accesses Windows C: drive files from Linux terminal |
| **Windows $\rightarrow$ Linux Path** | `\\wsl$\Ubuntu\home\abhishek\` | Accesses WSL files directly from Windows File Explorer |
| **Linux $\rightarrow$ VS Code / IDE** | `code .` | Opens the current directory in VS Code with WSL remote extension |

### B. Launching Windows Executables from Ubuntu
You can execute Windows CLI tools directly from bash:
```bash
# Open a file in Windows Notepad
notepad.exe myfile.txt

# Run PowerShell or CMD command from WSL
powershell.exe -Command "Get-Process"

# Open any Windows URL in default Windows browser
cmd.exe /c start https://github.com
```

### C. Port Forwarding & Localhost
WSL2 automatically bridges localhost to Windows:
* If you run a local web server, Jupyter Notebook (`localhost:8888`), or TensorBoard (`localhost:6006`) inside WSL, **you can open `http://localhost:6006` directly in your Windows browser**.

---

## 3. Essential Linux Terminal Commands Cheat Sheet

### A. Navigation & Directory Inspection

```bash
pwd                        # Print Working Directory (where am I?)
ls -lh                     # List files with human-readable sizes (MB, KB)
ls -la                     # List ALL files including hidden (starting with .)
tree -L 2                  # Show directory hierarchy up to 2 levels deep
cd ~                       # Go to your home directory (/home/username)
cd ..                      # Go up one directory level
cd -                       # Go back to the previous directory you were in
```

### B. File & Directory Management

```bash
mkdir -p path/to/deep/dir  # Create directory (including parent folders if missing)
touch file.txt             # Create an empty file or update timestamp
cp file.txt copy.txt       # Copy a file
cp -r src_dir/ dest_dir/   # Copy a folder recursively
mv old_name.py new_name.py # Rename or move a file/folder
rm file.txt                # Delete a file (Permanent! No Recycle Bin!)
rm -rf folder_name/        # Delete directory and everything in it (Use with caution!)
```

> [!CAUTION]
> In Linux, `rm -rf` deletes files permanently without moving them to a trash bin. Never run `sudo rm -rf /` or `rm -rf *` without verifying your current directory (`pwd`).

### C. Viewing & Searching Files

```bash
cat file.txt               # Print full content of file to terminal
head -n 20 file.txt        # View first 20 lines
tail -n 20 file.txt        # View last 20 lines
tail -f training.log       # Follow / stream updates to a file in real time
less file.txt              # Paginated viewer (Press 'q' to exit, '/' to search)
grep -rn "def train" .     # Search recursively for "def train" in current folder with line numbers
find . -name "*.py"        # Find all Python files in current directory and subdirectories
```

### D. System & Hardware Inspection

```bash
df -h                      # Show free disk space in human-readable format
free -h                    # Show total, used, and free RAM & Swap memory
uname -a                   # Show Linux kernel version
lsb_release -a             # Show Ubuntu distribution version
nvidia-smi                 # Inspect NVIDIA GPU usage, VRAM, and running processes
```

---

## 4. Ubuntu Package Management (`apt`)

`apt` (Advanced Package Tool) is the package manager for Ubuntu.

### A. The Golden Sequence for Installing Software
```bash
# 1. Always update package index first (downloads latest lists of software versions)
sudo apt update

# 2. Upgrade existing packages to latest versions (optional, recommended periodically)
sudo apt upgrade -y

# 3. Install desired software
sudo apt install -y build-essential curl wget git htop nvtop tmux jq
```

### B. Essential Utility Packages to Install

```bash
sudo apt install -y \
  build-essential \   # gcc, g++, make (needed for compiling C++/CUDA extensions)
  curl wget \         # Web download utilities
  git \               # Version control
  htop \              # Interactive CPU and RAM monitor
  nvtop \             # Beautiful interactive GPU monitor
  tmux \              # Terminal multiplexer (keeps jobs alive)
  tree \              # Directory visualizer
  jq \                # Command-line JSON parser
  ripgrep             # Blazing fast grep alternative (command: rg)
```

### C. Cleaning Up Space
WSL virtual disk space can grow over time. Clean apt caches regularly:
```bash
sudo apt autoremove -y     # Remove obsolete dependencies
sudo apt clean             # Clear downloaded .deb archive cache
```

---

## 5. Python & Virtual Environment Best Practices

### A. The Golden Python Rule in Ubuntu
> [!WARNING]
> **Never use `sudo pip install <package>` or install packages directly into `/usr/lib/python3/dist-packages`**.  
> Messing with system Python can break Ubuntu OS components. Always use a virtual environment.

### B. Standard `venv` Workflow
```bash
# 1. Install Python venv module for your Python version
sudo apt install -y python3.12-venv python3-pip

# 2. Navigate to your project directory (inside ~/ Linux filesystem)
cd ~/projects/my_project

# 3. Create isolated virtual environment
python3 -m venv .venv

# 4. Activate environment
source .venv/bin/activate
# Your terminal prompt will now prefix with (.venv)

# 5. Upgrade pip inside the venv
pip install --upgrade pip

# 6. Install requirements
pip install -r requirements.txt

# 7. When done working:
deactivate
```

### C. Cache Management (Hugging Face & Pip)
ML weights and datasets can rapidly fill your WSL disk.
* **Pip Cache**: `pip cache purge`
* **Hugging Face Cache**: Stored in `~/.cache/huggingface/hub/`
* To redirect Hugging Face cache to another location, add this to your `~/.bashrc`:
  ```bash
  export HF_HOME="/home/abhishek/.cache/huggingface"
  ```

---

## 6. GPU & CUDA Best Practices in WSL2

WSL2 supports native GPU acceleration through direct paravirtualized GPU passthrough.

### A. The #1 GPU Rule in WSL
> [!IMPORTANT]
> **DO NOT install Linux NVIDIA Display Drivers inside WSL!**  
> * Only install the standard NVIDIA Game Ready or Studio Driver on the **Windows Host**.  
> * WSL automatically mirrors the host driver into `/usr/lib/wsl/lib`.  
> * Inside WSL, you only install the **CUDA Toolkit** (compiler/libraries) and **PyTorch with CUDA**.

### B. Verifying GPU & CUDA Access
```bash
# 1. Check NVIDIA GPU visibility in WSL
nvidia-smi

# 2. Test PyTorch CUDA integration
python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

### C. Real-Time GPU Monitoring
Keep an eye on GPU VRAM and compute during model training:
```bash
# Monitor via watch (updates every 1 second)
watch -n 1 nvidia-smi

# OR run the interactive graphical tool (if nvtop is installed)
nvtop
```

---

## 7. Process & Job Management (Preventing Killed Tasks)

### A. Inspecting & Killing Processes

```bash
# Find process ID (PID) of running Python scripts
ps aux | grep python

# Monitor interactive system resource usage
htop

# Gracefully terminate a process by PID
kill <PID>

# Force kill a stubborn or frozen process
kill -9 <PID>

# Kill all processes matching a name
pkill -f train_gemma4
```

### B. Running Long Tasks with `tmux` (Terminal Multiplexer)
If you close your terminal or Windows Terminal restarts, all running scripts die. Use `tmux` to keep training running in the background.

```bash
# 1. Start a new named session
tmux new -s training

# 2. Inside tmux, activate environment and start training
source .venv/bin/activate
python train_gemma4_e4b.py

# 3. Detach from session (leave it running in background):
# Press: Ctrl + B, then press D

# 4. You can now close your terminal window completely!

# 5. Later, re-attach to the running session:
tmux attach -t training

# 6. List active sessions:
tmux ls

# 7. Kill a session when finished:
tmux kill-session -t training
```

---

## 8. Host Resource Management & `.wslconfig` Tuning

By default, WSL2 can consume up to 50% of total host RAM (or more) and all CPU cores, which can occasionally make Windows feel sluggish during heavy compilation or training.

### Setting up `.wslconfig` (Windows Host Side)
Create or edit a file named `.wslconfig` in your Windows User Profile directory (`C:\Users\<YourUsername>\.wslconfig`):

```ini
[wsl2]
# Limit memory allocated to WSL2 VM
memory=16GB

# Limit virtual processors
processors=8

# Limit swap space
swap=8GB

# Automatically release cached memory back to Windows
autoMemoryReclaim=gradual

# Enable localhost forwarding
localhostForwarding=true
```

### Managing WSL from Windows PowerShell (Run on Windows)
```powershell
# Check running WSL distributions and status
wsl --list --verbose

# Completely shut down WSL (frees all memory back to Windows)
wsl --shutdown

# Update WSL kernel to latest version
wsl --update

# Compact virtual hard disk to reclaim free disk space (WSL 2.0+)
wsl --manage Ubuntu --compact
```

### Freeing Cached Memory Inside Ubuntu
When Ubuntu caches large dataset files in RAM:
```bash
# Drop memory pagecache, dentries, and inodes
sudo sysctl -w vm.drop_caches=3
```

---

## 9. Customizing & Supercharging Your `~/.bashrc`

The `~/.bashrc` file executes every time you open a new bash terminal. Adding aliases and environment variables will save you thousands of keystrokes.

### Recommended Additions for `~/.bashrc`
Run `nano ~/.bashrc` (or edit via `code ~/.bashrc`) and append to the bottom:

```bash
# -------------------------------------------------------------
# Custom Aliases & Shortcuts
# -------------------------------------------------------------

# Navigation & Listing
alias ll='ls -laFh --color=auto'
alias ..='cd ..'
alias ...='cd ../..'
alias proj='cd ~/projects/unsloth'

# Quick Explorer & IDE Access
alias explorer='explorer.exe .'
alias code.='code .'

# Hardware & Process Monitoring
alias gpu='watch -n 1 nvidia-smi'
alias mygpu='nvtop'
alias mem='free -h'
alias disk='df -h'
alias dropcache='sudo sysctl -w vm.drop_caches=3'

# Git Shortcuts
alias gs='git status'
alias gp='git push'
alias gl='git log --oneline -n 10 --graph'
alias gd='git diff'

# Python & Environment
alias py='python3'
alias act='source .venv/bin/activate || source venv/bin/activate || source gemma_env/bin/activate'

# Safety Guards
alias rm='rm -i'    # Prompts for confirmation before deleting
alias cp='cp -i'    # Prompts before overwriting
alias mv='mv -i'    # Prompts before overwriting
```

After modifying `~/.bashrc`, reload it instantly:
```bash
source ~/.bashrc
```

---

## 10. Git, Line Endings & SSH Configuration

### A. Avoiding Line Ending Mismatches (`CRLF` vs `LF`)
Windows uses Carriage Return + Line Feed (`\r\n`), while Linux uses Line Feed (`\n`). This can cause git diff noise or bash script execution errors (`\r: command not found`).

Configure git inside WSL:
```bash
git config --global core.autocrlf input
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### B. Setting up SSH Key for GitHub
```bash
# 1. Generate SSH key (press Enter to accept default path)
ssh-keygen -t ed25519 -C "your.email@example.com"

# 2. Start ssh-agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# 3. Print public key and copy it to GitHub Settings -> SSH Keys
cat ~/.ssh/id_ed25519.pub
```

---

## 11. Troubleshooting Common WSL & Ubuntu Errors

| Issue / Error | Root Cause | Solution |
| :--- | :--- | :--- |
| `bash: ./script.sh: /bin/bash^M: bad interpreter` | Script was saved with Windows CRLF line endings | Run `dos2unix script.sh` or `sed -i -e 's/\r$//' script.sh` |
| `Permission denied` when running a script | File lacks executable bit | Run `chmod +x script.sh` |
| `pip install` extremely slow / freezing | Project is located in `/mnt/c/` instead of `~/` | Move project directory to `/home/username/` |
| `nvidia-smi: command not found` | NVIDIA host driver not installed on Windows or outdated WSL | Update NVIDIA Windows driver & run `wsl --update` in PowerShell |
| `vmmem` consuming all Windows RAM | Linux caching disk files in virtual memory | Set memory limit in `.wslconfig` and run `wsl --shutdown` periodically |
| `CUDA out of memory` during training | Batch size or sequence length too high for VRAM | Reduce `max_seq_length` to 1024, set batch size to 1, use `adamw_8bit` |
| `Address already in use` when starting server | Previous instance still running in background | Run `lsof -i :<port>` or `fuser -k <port>/tcp` to kill process |

---

## 12. Quick Reference Summary Table

```
┌──────────────────────────────┬────────────────────────────────────────────────────────┐
│ Goal                         │ Recommended Command                                    │
├──────────────────────────────┼────────────────────────────────────────────────────────┤
│ Open current folder in Win   │ explorer.exe .                                         │
│ Monitor GPU in real-time     │ nvtop  OR  watch -n 1 nvidia-smi                       │
│ Keep long script running     │ tmux new -s run1  (Detach: Ctrl+B then D)              │
│ Check RAM usage              │ free -h                                                │
│ Check Disk usage             │ df -h                                                  │
│ Find Python processes        │ ps aux | grep python                                   │
│ Update system software       │ sudo apt update && sudo apt upgrade -y                 │
│ Reload terminal settings     │ source ~/.bashrc                                       │
│ Shutdown WSL from Windows    │ wsl --shutdown (in PowerShell)                         │
└──────────────────────────────┴────────────────────────────────────────────────────────┘
```

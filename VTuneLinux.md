# Installing VTune on native linux

- Open challenges:
  - How to do this on  Microsoft's WSL2 kernel?
  - How to install just basic `perf` on Microsoft's WSL2 kernel?

## Prerequisites

 - You need an Intel CPU (any - even AVX-512 is supported).
 - Root access
 
## Intel VTune via apt

Add Intel's oneapi to apt
```bash
wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | sudo gpg --dearmor -o /usr/share/keyrings/intel-oneapi-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/intel-oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/intel-oneapi.list
```
Install
```bash
sudo apt install intel-oneapi-vtune
```

To activate it, call
```bash
source /opt/intel/oneapi/setvars.sh
```
or make it permanently, by adding
```bash
# Source Intel oneAPI environment silently
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
fi
```
to your `$USER/.bashrc`.

## Install drivers for hardware access
```bash
cd /opt/intel/oneapi/vtune/latest/sepdk/src
sudo ./build-driver
sudo ./insmod-sep -r  # Unloads old modules
sudo ./insmod-sep     # Loads sampling drivers
```
## Make stuff available to current user

### Ensure Secure Boot is disabled
```bash
sudo apt install mokutil
mokutil --sb-state
```
If present, go to BIOS/UEFI and disable Secure Boot
1. Reboot your system
2. Enter UEFI Setup (usually F2 or Del)
3. Navigate to Boot/Security tab
4. Disable Secure Boot
5. Save and reboot`

### Add and update `vtune` group

```bash
sudo groupadd vtune
sudo usermod -aG vtune $USER
```
Investigate which devices to allow for the `vtune` group
```bash
sudo strace -e trace=openat vtune -collect memory-access -result-dir vtune_data ./baseline 2>&1 | grep /dev/
```
Output depends on your hardware and kernel. Add rights to group
```bash
chgrp vtune /dev/sep5 /dev/sep5_* /dev/sep5/* /dev/pax /dev/apwr_driver_char_dev /dev/socperf3/c
sudo chmod 660 /dev/sep5 /dev/sep5_* /dev/sep5/* /dev/pax /dev/apwr_driver_char_dev /dev/socperf3/c
```
Create rules for vtune
```bash
sudo tee /etc/udev/rules.d/99-vtune.rules > /dev/null <<EOF
KERNEL=="sep5", MODE="0660", GROUP="vtune"
KERNEL=="sep5_*", MODE="0660", GROUP="vtune"
KERNEL=="pax", MODE="0660", GROUP="vtune"
KERNEL=="socperf3*", MODE="0660", GROUP="vtune"
KERNEL=="apwr_driver_char_dev", MODE="0660", GROUP="vtune"
EOF
```
Update rules
```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### Disable kernel lockdown
Check if kernel lockdown is enabled
```
cat /sys/kernel/security/lockdown
```
If you see something like,
```bash
[none] integrity confidentiality
```
you need to update your boot loader
```bash
sudo nano /etc/default/grub
```
and add `lockdown=none` to the defaults, e.g.
```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash lockdown=none"
```

### Optional thermal information
To allow VTune to access thermal information
```bash
echo 'kernel.perf_event_paranoid=0' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```
To make it permanent and load at boot
```bash
sudo tee /etc/systemd/system/vtune-sampling.service > /dev/null <<'EOF'
[Unit]
Description=Intel VTune Sampling Driver
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/opt/intel/oneapi/vtune/latest/sepdk/src/insmod-sep
ExecStartPost=/bin/sh -c 'chgrp vtune /dev/sep5 && chmod 660 /dev/sep5'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
```

# PC Related Things

## Installed things

```bash
sudo apt install tmux -y
sudo apt install nvtop -y
sudo apt install htop -y
sudo apt install tree -y
sudo apt install htop -y

sudo apt install ros-humble-librealsense2*
sudo apt install ros-humble-realsense2-camera
```


## Other Modifications to the PC

### Make CPU performance persistent after reboot


```bash
# to check the current CPU mode
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference
```

```bash
sudo apt update && sudo apt install linux-tools-$(uname -r)
```

```bash
sudo nano /etc/systemd/system/cpu-performance.service
```

```bash
[Unit]
Description=Set CPU Governor to Performance
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/bin/cpupower frequency-set -g performance
ExecStartPost=/bin/sh -c "echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference"
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now cpu-performance.service
```

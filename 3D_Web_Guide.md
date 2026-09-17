# 🚁 3D Swarm Web Interface Guide

## ✅ **Problem Fixed!**

The 3D swarm mode (`ppo_3d`) is now working in the web interface! I fixed the missing `obstacles` and `covered` attributes in the 3D environment.

## 🌐 **How to Access 3D Mode**

1. **Open web interface:** http://127.0.0.1:8001

2. **Select Mode:** In the sidebar, change the dropdown from "Known map" to **"3D learned search (PPO) -- altitude + damage detection"**

3. **Choose Location:** Pick a disaster location:
   - Amatrice, Italy (2016 earthquake)
   - Lahaina, Maui (2023 wildfire)  
   - Christchurch, NZ (2011 earthquake)
   - Kahramanmaras, Turkey (2023 earthquake)

4. **Deploy Swarm:** Click the green "Deploy Swarm" button

## 🎮 **3D Mode Features**

### **Altitude-Aware Search:**
- Drones start at maximum altitude (altitude 4)
- They learn to descend to get better sensor readings
- Higher altitude = larger sensing radius but lower confidence
- Lower altitude = smaller radius but higher damage detection confidence

### **Terrain Visualization:**
- Map shows terrain height as shading (darker = higher terrain)
- Drones must navigate around/over terrain obstacles
- Real geographic locations with elevation data

### **Damage Detection:**
- Drones detect structural damage as they explore
- Different severity levels: Minor (yellow), Major (orange), Destroyed (red)
- Alarm system triggers for significant damage discoveries
- Priority zones highlight areas needing immediate attention

### **3D Stats Panel:**
When in 3D mode, you'll see additional stats:
- Explored percentage
- Confirmed damage counts by severity
- Alarm notifications
- Priority zone rankings

## 🔧 **Technical Details**

### **What I Fixed:**
1. **Added `obstacles` attribute** to 3D environment (terrain height > 0)
2. **Added `covered` attribute** tracking observed cells
3. **Updated sensing logic** to maintain coverage state
4. **Ensured compatibility** with 2D movement policies

### **3D vs 2D Differences:**
- **2D:** Flat grid with known/unknown obstacles, coverage tracking
- **3D:** Terrain heights, altitude navigation, damage assessment, discovery mode

### **PPO Policy:**
The 3D mode uses a pre-trained PPO (Proximal Policy Optimization) network that learned:
- Optimal altitude selection for different tasks
- Efficient exploration patterns
- Damage detection strategies
- Coordination between multiple drones

## 🚀 **Try It Now!**

1. Open http://127.0.0.1:8001
2. Select "3D learned search (PPO)" mode  
3. Pick a disaster location
4. Watch the AI-controlled drones explore in 3D!

The drones will intelligently:
- Adjust altitude for optimal sensing
- Discover terrain and damage
- Coordinate to cover the search area efficiently
- Generate alerts for critical findings

Your 3D simulation is now fully functional! 🎉
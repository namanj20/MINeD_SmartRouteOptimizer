# **Smart Route Optimizer**

## **MINeD Hackathon 2025 Project**

An advanced algorithm-driven route optimization tool designed to enhance last-mile delivery efficiency. It ensures **on-time shipments**, **optimal vehicle utilization**, and **cost-effective logistics planning**. 

---

## **Key Features**
- **Smart Vehicle Assignment**: Prioritizes **3W and 4W-EV vehicles** before using standard 4W vehicles.
- **Optimized Route Planning**: Uses **clustering + graph algorithms** for the shortest delivery paths.
- **Time Slot Enforcement**: **Shipments must be delivered within the assigned time slot** or the trip is invalid.
- **Live Vehicle Tracking**: Monitors **vehicle departure, delivery success, and return**.
- **Space & Time Utilization Tracking**: Ensures **>50% vehicle capacity utilization** and efficient time slot scheduling.
- **Multi-Slot Assignments**: Dynamically groups deliveries across overlapping time slots to optimize space.
- **Interactive Visualization**: Displays optimized delivery routes on a map.

---

## **How It Works**
1. **Load Shipment Data**
- Input dataset contains **shipment coordinates, delivery time slots, and vehicle constraints**.

2. **Clustering & Route Optimization**
- **K-Means Clustering** groups nearby shipments.
- **Dijkstra's Algorithm** finds the shortest route.

3. **Smart Vehicle Assignment**
- Assigns **3W & 4W-EV first**, followed by **4W** (if necessary).
- Ensures **>50% capacity utilization** before dispatching.

4. **Time & Space Optimization**
- Computes **trip time** using:
  - **5 min/km** travel time.
  - **10 min per shipment drop-off.**
- Validates **time slot adherence** (failure invalidates the trip).

5. **Tracking & Logging**
- Records **successful deliveries and vehicle usage logs**.
- Outputs structured **CSV report** for analysis.

---

## **Output CSV Format**
| **Column Name**     | **Description** |
|----------------|-------------|
| **TRIP_ID** | Unique trip identifier |
| **Shipment ID** | Shipment assigned to the trip |
| **Latitude, Longitude** | Delivery location |
| **TIME SLOT** | Scheduled delivery window |
| **Shipments** | Number of shipments in a trip |
| **MST_DIST** | Distance covered by the vehicle |
| **TRIP_TIME** | Estimated total time for the trip |
| **Vehicle_Type** | Assigned vehicle for the trip |
| **CAPACITY_UTI** | Vehicle space utilization |
| **TIME_UTI** | Time efficiency ratio |
| **COV_UTI** | Distance utilization |

---

## **Installation & Setup**
### **Requirements**
- **Python 3.x**
- **Required Libraries:**
  ```bash
  pip install pandas numpy geopy scikit-learn networkx tkinter folium haversine openpyxl
  ```

### **Running the Application**
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/OptiShip.git
   cd OptiShip
   ```
2. **Run the Smart Route Optimizer application:**
   ```bash
   python hackathon.py
   ```
3. **Upload shipment data in CSV format and start optimization.**

---

## **How Else Can You Use This?**
- **E-commerce Logistics**: Optimize **Amazon/Flipkart-style** last-mile deliveries.
- **Fleet Management**: Smart route assignment for **delivery services & food delivery apps**.
- **Corporate Dispatch**: Automate **company courier & product deliveries**.

---

## **Future Enhancements**
- **Real-Time Traffic Optimization** using live API integration.  
- **Machine Learning Model** for dynamic delivery predictions.  
- **Mobile App Interface** for driver tracking & real-time routing.  

---

## **Credits**
Developed for **MINeD Hackathon 2025** by **Team InHuMaNs [Mirav Patel, Naman Jain, Hrishita Patni]**.

**Let's revolutionize delivery logistics!**

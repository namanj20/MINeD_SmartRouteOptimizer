# MINDeD Smart Route Optimizer

## Introduction

Smart Route Optimizer is an advanced route optimization solution designed to enhance logistics efficiency. This application provides a user-friendly interface for optimizing delivery routes based on various constraints such as vehicle capacity, delivery time slots, and geographical locations.

## Features

- User-friendly GUI built with tkinter
- File input for delivery data
- Advanced route optimization algorithm
- Interactive map visualization of optimized routes
- Exportable results in Excel format

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Dependencies

Install the required packages using the following command:


pip install pandas numpy scikit-learn haversine openpyxl folium tkinter


## Usage

1. Clone the repository or download the source code.
2. Navigate to the project directory.
3. Run the application:


python smart_route_optimizer.py


4. Use the GUI to select your input CSV file containing delivery data.
5. Click "Start Optimization" to process the data.
6. View the optimized routes in the table.
7. Use the "View Trip" button to visualize specific routes on a map.
8. Download the results using the "Download Results" button.

## Input File Format

The input CSV file should contain the following columns:
- Shipment ID
- Latitude
- Longitude
- Delivery Timeslot

## Output

The application generates an Excel file named "formatted_trip_details.xlsx" containing the optimized route information.

## Contributing

Contributions to the Smart Route Optimizer project are welcome. Please feel free to submit pull requests or open issues to suggest improvements or report bugs.

## License

[Include your license information here]

## Contact

[Your contact information or project maintainer's contact]

## Acknowledgments

- This project uses several open-source libraries including pandas, numpy, scikit-learn, and folium.
- Special thanks to the contributors and maintainers of these libraries.

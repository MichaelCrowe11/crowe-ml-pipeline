# Installation Instructions for the Universal Fungal Intelligence System

## Prerequisites

Before installing the Universal Fungal Intelligence System, ensure that you have the following software installed on your machine:

- Python 3.7 or higher
- pip (Python package installer)
- Git (optional, for cloning the repository)

## Installation Steps

1. **Clone the Repository**

   Open your terminal and run the following command to clone the repository:

   ```
   git clone https://github.com/yourusername/universal-fungal-intelligence-system.git
   ```

   Replace `yourusername` with your GitHub username.

2. **Navigate to the Project Directory**

   Change your working directory to the project folder:

   ```
   cd universal-fungal-intelligence-system
   ```

3. **Create a Virtual Environment (Optional but Recommended)**

   It is recommended to create a virtual environment to manage dependencies. You can create a virtual environment using the following command:

   ```
   python -m venv venv
   ```

   Activate the virtual environment:

   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. **Install Dependencies**

   Install the required Python packages using pip:

   ```
   pip install -r requirements.txt
   ```

   If you are using Docker, you can skip this step and proceed to the Docker installation.

5. **Docker Installation (Optional)**

   If you prefer to run the application using Docker, ensure that Docker is installed on your machine. Then, build the Docker image with the following command:

   ```
   docker build -t universal-fungal-intelligence-system .
   ```

   After building the image, you can run the application using:

   ```
   docker-compose up
   ```

6. **Run the Application**

   After installing the dependencies, you can run the application using the following command:

   ```
   python src/main.py
   ```

   This will start the Universal Fungal Intelligence System.

## Additional Information

For more details on usage and configuration, please refer to the [Usage Documentation](usage.md) and [Development Guidelines](development.md).
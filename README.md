# AI Acne Identification & Treatment Recommendation System

An intelligent system that identifies different types of acne on facial images and provides personalized treatment recommendations.

## Features

- **Real-time Acne Detection**: Identifies 6 types of acne lesions (blackheads, dark spots, nodules, papules, pustules, whiteheads)
- **Severity Assessment**: Evaluates acne severity based on lesion count and distribution
- **Treatment Recommendations**: Provides personalized treatment suggestions based on detected acne types
- **Progress Tracking**: Allows users to monitor their skin condition over time
- **Educational Content**: Includes information about different acne types and treatments

## Dataset

This project uses the [Acne Object Detection Dataset](https://universe.roboflow.com/kritsakorn/acne-kbm0q) from Roboflow, which contains:
- 6 acne classes with bounding box annotations
- High-quality facial images
- Diverse skin tones and lighting conditions

## Project Structure

```
belle/
├── app/                    # Main application code
│   ├── models/            # ML model files
│   ├── api/               # FastAPI endpoints
│   ├── services/          # Business logic
│   └── utils/             # Utility functions
├── data/                  # Dataset and processed data
├── notebooks/             # Jupyter notebooks for analysis
├── tests/                 # Test files
├── static/                # Frontend assets
├── templates/             # HTML templates
└── requirements.txt       # Python dependencies
```

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Download the dataset from Roboflow
2. Train the model:
   ```bash
   python app/models/train.py
   ```
3. Start the web application:
   ```bash
   uvicorn app.api.main:app --reload
   ```
4. Open your browser to `http://localhost:8000`

## API Endpoints

- `POST /upload` - Upload facial image for analysis
- `GET /results/{session_id}` - Get analysis results
- `GET /treatments` - Get treatment recommendations
- `GET /health` - Health check

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

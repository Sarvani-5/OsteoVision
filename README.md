# 🦴 Knee Osteoarthritis Severity Classifier

## 🚀 Advanced AI-Powered Radiology Assistant

A cutting-edge Flask web application that uses deep learning (ResNet50) to classify knee X-ray images and assess osteoarthritis severity with Grad-CAM visualization. Perfect for medical research, education, and interview projects!

## ✨ Features

- **🤖 AI-Powered Classification**: ResNet50 deep learning model for accurate knee OA assessment
- **🔍 Grad-CAM Visualization**: See exactly where the AI focuses its attention (red heatmap areas)
- **🎨 Beautiful Dark Mode UI**: Modern, responsive design with red accents and smooth animations
- **📱 Mobile Responsive**: Works perfectly on all devices
- **⚡ Real-time Analysis**: Fast processing with confidence scores and detailed breakdowns
- **🔄 Drag & Drop**: Intuitive file upload with visual feedback

## 🏥 Medical Classification

The system classifies knee X-ray images into 5 KL (Kellgren-Lawrence) grades:

| Grade | Severity | Description                                          | Color Code  |
| ----- | -------- | ---------------------------------------------------- | ----------- |
| 0     | Normal   | No signs of osteoarthritis                           | 🟢 Green    |
| 1     | Doubtful | Possible minimal osteophytes                         | 🟡 Yellow   |
| 2     | Mild     | Definite osteophytes, possible joint space narrowing | 🟠 Orange   |
| 3     | Moderate | Multiple osteophytes, definite joint space narrowing | 🔴 Red      |
| 4     | Severe   | Large osteophytes, severe joint space narrowing      | 💀 Dark Red |

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- Your ResNet50.keras model file in the project directory

### 1. Clone/Download Project

```bash
# Navigate to your project directory
cd your-project-folder
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Model File

Ensure `ResNet50.keras` is in your project root directory.

### 4. Run the Application

```bash
python stroke.py
```

### 5. Access the Application

Open your browser and go to: `http://localhost:5000`

## 📁 Project Structure

```
project/
├── stroke.py              # Main Flask application
├── ResNet50.keras         # Your trained model
├── templates/
│   └── index.html        # Beautiful UI template
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🎯 How It Works

1. **Image Upload**: Drag & drop or click to upload knee X-ray images
2. **AI Processing**: ResNet50 model analyzes the image and generates predictions
3. **Grad-CAM Generation**: Creates heatmap showing AI's focus areas (red = high attention)
4. **Results Display**: Shows classification, confidence scores, and visualizations
5. **Detailed Analysis**: Confidence distribution across all severity classes

## 🔬 Technical Details

- **Backend**: Flask web framework
- **AI Model**: ResNet50 with custom training for knee OA classification
- **Image Processing**: OpenCV for image manipulation and preprocessing
- **Grad-CAM**: Gradient-weighted Class Activation Mapping for explainable AI
- **Frontend**: Modern HTML5, CSS3, and JavaScript with responsive design
- **Image Format Support**: JPG, JPEG, PNG (Max 16MB)

## 🎨 UI/UX Features

- **Dark Mode**: Professional medical interface with red accent colors
- **Smooth Animations**: Hover effects, loading spinners, and success animations
- **Responsive Grid**: Adaptive layout for all screen sizes
- **Interactive Elements**: Hover effects, drag & drop, and visual feedback
- **Professional Typography**: Clean, readable fonts with proper hierarchy

## 🚨 Important Notes

- **Research Purpose Only**: This tool is designed for educational and research purposes
- **Medical Disclaimer**: Always consult qualified medical professionals for actual diagnosis
- **Model Requirements**: Ensure your ResNet50 model is trained for knee OA classification
- **Image Quality**: Higher quality X-ray images yield better results

## 🔧 Customization

### Changing Model

- Replace `ResNet50.keras` with your custom model
- Update the `load_keras_model()` function in `stroke.py`
- Adjust preprocessing parameters if needed

### UI Modifications

- Edit `templates/index.html` for frontend changes
- Modify CSS variables for color schemes
- Add new features in the JavaScript section

### Adding New Classes

- Update the `class_info` dictionary in `stroke.py`
- Modify the confidence chart in the HTML template
- Adjust color schemes and descriptions

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Error**

   - Ensure `ResNet50.keras` is in the correct directory
   - Check model file integrity
   - Verify TensorFlow version compatibility

2. **Image Processing Error**

   - Check image format (JPG, JPEG, PNG only)
   - Ensure image size is reasonable (< 16MB)
   - Verify image file isn't corrupted

3. **Port Already in Use**
   - Change port in `stroke.py`: `app.run(port=5001)`
   - Or kill existing process using port 5000

### Performance Tips

- Use GPU if available (TensorFlow will auto-detect)
- Optimize image size before upload
- Close other resource-intensive applications

## 📊 Performance Metrics

- **Processing Time**: Typically 2-5 seconds per image
- **Accuracy**: Depends on your trained model quality
- **Memory Usage**: ~2-4GB RAM during processing
- **Supported Formats**: JPG, JPEG, PNG
- **Max File Size**: 16MB

## 🌟 Why This Project Stands Out

1. **Medical AI Application**: Real-world healthcare technology
2. **Explainable AI**: Grad-CAM visualization for transparency
3. **Professional UI/UX**: Interview-ready, production-quality interface
4. **Modern Tech Stack**: Latest Flask, TensorFlow, and frontend technologies
5. **Comprehensive Documentation**: Easy setup and customization
6. **Responsive Design**: Works on all devices and screen sizes

## 📈 Future Enhancements

- [ ] Batch processing for multiple images
- [ ] Export results to PDF reports
- [ ] Integration with PACS systems
- [ ] Real-time video analysis
- [ ] Multi-language support
- [ ] User authentication and history
- [ ] API endpoints for external integration

## 🤝 Contributing

Feel free to contribute improvements, bug fixes, or new features! This project is perfect for:

- Medical AI researchers
- Computer vision enthusiasts
- Web developers
- Healthcare technology students
- Interview portfolio projects

## 📄 License

This project is for educational and research purposes. Please ensure compliance with your institution's policies and medical regulations.

## 🆘 Support

For issues or questions:

1. Check the troubleshooting section
2. Verify your setup matches the requirements
3. Ensure your model file is compatible
4. Check console logs for detailed error messages

---

**🎯 Perfect for Interviews**: This project demonstrates advanced AI, medical technology, modern web development, and professional UI/UX design - everything interviewers love to see!

**🏥 Medical AI Innovation**: Showcases cutting-edge technology in healthcare, making it highly relevant and impressive.



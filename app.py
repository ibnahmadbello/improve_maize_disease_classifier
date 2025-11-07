# # app_fixed_tf.py
# import os
# import streamlit as st
# import numpy as np
# from PIL import Image
# import joblib
# import pickle
# import pandas as pd
# import matplotlib.pyplot as plt

# # ===== CRITICAL: FORCE CPU AND SUPPRESS TENSORFLOW ERRORS =====
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Suppress TensorFlow logs
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable problematic optimizations

# # Import TensorFlow with error suppression
# try:
#     import tensorflow as tf
#     # Force CPU usage
#     tf.config.set_visible_devices([], 'GPU')
#     from tensorflow.keras.applications.resnet50 import preprocess_input
#     from tensorflow.keras.preprocessing import image
#     TENSORFLOW_AVAILABLE = True
# except ImportError:
#     st.error("TensorFlow not available")
#     TENSORFLOW_AVAILABLE = False
# except Exception as e:
#     st.error(f"TensorFlow initialization failed: {e}")
#     TENSORFLOW_AVAILABLE = False

# # Set page config
# st.set_page_config(page_title="Maize Disease Detector", layout="wide", page_icon="🌽")
# st.title("🌽 Maize Leaf Disease Classification")
# st.markdown("Upload an image of a maize leaf for automated disease diagnosis")

# @st.cache_resource
# def load_assets():
#     try:
#         required_files = ['maize_rf_model.pkl', 'class_names.pkl']
#         missing_files = [f for f in required_files if not os.path.exists(f)]
        
#         if missing_files:
#             st.error(f"❌ Missing files: {', '.join(missing_files)}")
#             return None, None, None
        
#         # Load Random Forest model
#         rf_model = joblib.load('maize_rf_model.pkl')
        
#         # Load class names
#         with open('class_names.pkl', 'rb') as f:
#             class_names = pickle.load(f)
            
#         st.sidebar.success("✅ Models loaded successfully!")
#         return rf_model, class_names, None
        
#     except Exception as e:
#         st.sidebar.error(f"❌ Error loading models: {str(e)}")
#         return None, None, None

# def extract_features_cnn(img):
#     """Extract features using ResNet50 CNN (2048 features)"""
#     try:
#         # Load the feature extractor model safely
#         if 'feature_extractor' not in st.session_state:
#             with st.spinner('Loading CNN feature extractor...'):
#                 st.session_state.feature_extractor = tf.keras.models.load_model(
#                     'feature_extractor.h5', 
#                     compile=False,
#                     safe_mode=False  # Bypass safety checks if needed
#                 )
        
#         model = st.session_state.feature_extractor
        
#         # Preprocess image
#         img = img.resize((224, 224))
#         x = image.img_to_array(img)
#         x = np.expand_dims(x, axis=0)
#         x = preprocess_input(x)
        
#         # Extract features
#         features = model.predict(x, verbose=0)
#         return features.flatten()
        
#     except Exception as e:
#         st.error(f"CNN feature extraction failed: {e}")
#         return None

# # Sidebar
# with st.sidebar:
#     st.header("ℹ️ System Info")
#     if TENSORFLOW_AVAILABLE:
#         st.success(f"TensorFlow: {tf.__version__}")
#         st.info("Using CNN feature extraction (2048 features)")
#     else:
#         st.warning("TensorFlow not available")
#         st.error("Cannot extract CNN features")

# # Main app
# uploaded_file = st.file_uploader("Choose a maize leaf image...", 
#                                 type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:
#     rf_model, class_names, _ = load_assets()
    
#     if rf_model is None:
#         st.error("Please ensure model files exist in the current directory")
#         st.stop()
    
#     try:
#         img = Image.open(uploaded_file).convert('RGB')
        
#         # Display image
#         col1, col2 = st.columns([1, 1])
#         with col1:
#             st.image(img, caption="Uploaded Image", use_column_width=True)
#             st.caption(f"Size: {img.size[0]}x{img.size[1]} pixels")
        
#         with col2:
#             if not TENSORFLOW_AVAILABLE:
#                 st.error("❌ TensorFlow not available. Cannot extract features.")
#                 st.info("Please install TensorFlow or use the original training environment")
#                 st.stop()
            
#             with st.spinner('🔬 Extracting CNN features...'):
#                 features = extract_features_cnn(img)
                
#                 if features is None:
#                     st.error("Feature extraction failed")
#                     st.stop()
                
#                 st.success(f"✓ Extracted {len(features)} features")
                
#                 # Verify feature count matches Random Forest expectations
#                 if hasattr(rf_model, 'n_features_in_'):
#                     expected_features = rf_model.n_features_in_
#                     if len(features) != expected_features:
#                         st.error(f"❌ Feature mismatch: Expected {expected_features}, got {len(features)}")
#                         st.stop()
                
#                 # Make prediction
#                 with st.spinner('🤖 Making prediction...'):
#                     try:
#                         probabilities = rf_model.predict_proba([features])[0]
#                         predicted_class = class_names[np.argmax(probabilities)]
#                         confidence = np.max(probabilities) * 100
                        
#                         # Display results
#                         st.subheader("📋 Diagnosis Results")
                        
#                         if predicted_class != 'Healthy':
#                             st.error(f"**🦠 Detected:** {predicted_class}")
#                             st.warning(f"**🎯 Confidence:** {confidence:.1f}%")
#                         else:
#                             st.success(f"**✅ Healthy**")
#                             st.success(f"**🎯 Confidence:** {confidence:.1f}%")
                        
#                         # Probability table
#                         st.subheader("📊 Probabilities")
#                         prob_data = []
#                         for cls, prob in zip(class_names, probabilities):
#                             prob_data.append({
#                                 'Disease': cls,
#                                 'Probability': f"{prob:.1%}",
#                                 'Value': prob
#                             })
                        
#                         prob_df = pd.DataFrame(prob_data)
#                         st.dataframe(prob_df[['Disease', 'Probability']], 
#                                    hide_index=True, use_container_width=True)
                        
#                         # Visual chart
#                         fig, ax = plt.subplots(figsize=(10, 4))
#                         colors = ['red' if 'Healthy' not in cls else 'green' for cls in class_names]
#                         bars = ax.bar(class_names, probabilities, color=colors, alpha=0.7)
#                         ax.set_title('Disease Probability Distribution')
#                         ax.set_ylabel('Probability')
#                         ax.set_ylim(0, 1)
#                         plt.xticks(rotation=45)
                        
#                         for bar, p in zip(bars, probabilities):
#                             ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
#                                    f'{p:.1%}', ha='center', va='bottom', fontsize=9)
                        
#                         st.pyplot(fig)
                        
#                     except Exception as e:
#                         st.error(f"❌ Prediction failed: {str(e)}")
                        
#     except Exception as e:
#         st.error(f"❌ Error processing image: {str(e)}")

# # Footer
# st.markdown("---")
# st.caption("🔧 Using original CNN features | ⚠️ Requires TensorFlow")


# # app_final.py
# import os
# import streamlit as st
# import numpy as np
# from PIL import Image
# import joblib
# import pickle
# import pandas as pd
# import matplotlib.pyplot as plt

# # ===== TENSORFLOW CONFIGURATION =====
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Suppress TensorFlow logs

# try:
#     import tensorflow as tf
#     tf.config.set_visible_devices([], 'GPU')  # Force CPU usage
#     from tensorflow.keras.applications.resnet50 import preprocess_input
#     from tensorflow.keras.preprocessing import image
#     TENSORFLOW_AVAILABLE = True
# except Exception as e:
#     st.error(f"TensorFlow initialization failed: {e}")
#     TENSORFLOW_AVAILABLE = False

# # ===== STREAMLIT APP CONFIG =====
# st.set_page_config(
#     page_title="Maize Disease Detector", 
#     layout="wide", 
#     page_icon="🌽",
#     initial_sidebar_state="expanded"
# )

# st.title("🌽 Maize Leaf Disease Classification")
# st.markdown("Upload an image of a maize leaf for automated disease diagnosis")

# # ===== LOAD MODELS =====
# @st.cache_resource
# def load_models():
#     """Load all required models with error handling"""
#     try:
#         # Check for required files
#         required_files = ['maize_rf_model.pkl', 'feature_extractor.h5', 'class_names.pkl']
#         missing_files = [f for f in required_files if not os.path.exists(f)]
        
#         if missing_files:
#             return None, None, None, f"Missing files: {', '.join(missing_files)}"
        
#         # Load Random Forest model
#         rf_model = joblib.load('maize_rf_model.pkl')
        
#         # Load feature extractor
#         feature_extractor = tf.keras.models.load_model('feature_extractor.h5', compile=False)
        
#         # Load class names
#         with open('class_names.pkl', 'rb') as f:
#             class_names = pickle.load(f)
        
#         return rf_model, feature_extractor, class_names, None
        
#     except Exception as e:
#         return None, None, None, f"Error loading models: {str(e)}"

# # ===== FEATURE EXTRACTION =====
# def extract_cnn_features(img, model):
#     """Extract features using CNN model"""
#     try:
#         # Preprocess image
#         img = img.resize((224, 224))
#         x = image.img_to_array(img)
#         x = np.expand_dims(x, axis=0)
#         x = preprocess_input(x)
        
#         # Extract features
#         features = model.predict(x, verbose=0)
#         return features.flatten()
#     except Exception as e:
#         st.error(f"Feature extraction error: {e}")
#         return None

# # ===== SIDEBAR =====
# with st.sidebar:
#     st.header("ℹ️ About")
#     st.info("""
#     **Detects maize leaf diseases:**
#     - 🦠 Common Rust
#     - 🍂 Gray Leaf Spot 
#     - 🍁 Northern Leaf Blight
#     - ✅ Healthy leaves
    
#     **Model Info:**
#     - CNN Feature Extraction (ResNet50)
#     - Random Forest Classification
#     - 2048-dimensional features
#     """)
    
#     st.header("📊 Model Status")
#     if TENSORFLOW_AVAILABLE:
#         st.success("✅ TensorFlow: Active")
#         st.code(f"Version: {tf.__version__}")
#     else:
#         st.error("❌ TensorFlow: Not Available")
    
#     # File status
#     st.subheader("📁 File Status")
#     for file in ['maize_rf_model.pkl', 'feature_extractor.h5', 'class_names.pkl']:
#         if os.path.exists(file):
#             size_mb = os.path.getsize(file) / (1024 * 1024)
#             st.success(f"✅ {file} ({size_mb:.1f} MB)")
#         else:
#             st.error(f"❌ {file} (Missing)")

# # ===== MAIN APP =====
# # Load models
# rf_model, feature_extractor, class_names, error_msg = load_models()

# if error_msg:
#     st.error(f"❌ {error_msg}")
#     st.info("""
#     **Required files in current directory:**
#     1. `maize_rf_model.pkl` - Random Forest model
#     2. `feature_extractor.h5` - CNN feature extractor  
#     3. `class_names.pkl` - Class labels
#     """)
#     st.stop()

# if not TENSORFLOW_AVAILABLE:
#     st.error("❌ TensorFlow is required but not available")
#     st.stop()

# # File uploader
# uploaded_file = st.file_uploader(
#     "📤 Choose a maize leaf image...", 
#     type=["jpg", "jpeg", "png"],
#     help="Select a clear, well-lit image of a maize leaf"
# )

# if uploaded_file is not None:
#     try:
#         # Load and display image
#         img = Image.open(uploaded_file).convert('RGB')
        
#         col1, col2 = st.columns([1, 1])
        
#         with col1:
#             st.image(img, caption="Uploaded Image", use_column_width=True)
#             st.caption(f"**Dimensions:** {img.size[0]}×{img.size[1]} pixels")
#             st.caption(f"**Format:** {img.format}")
        
#         with col2:
#             # Analysis section
#             st.subheader("🔍 Analysis Results")
            
#             with st.spinner('Extracting features...'):
#                 features = extract_cnn_features(img, feature_extractor)
                
#                 if features is None:
#                     st.error("Feature extraction failed")
#                     st.stop()
                
#                 # Verify feature dimensions
#                 expected_features = getattr(rf_model, 'n_features_in_', 2048)
#                 if len(features) != expected_features:
#                     st.error(f"Feature dimension mismatch: Expected {expected_features}, got {len(features)}")
#                     st.stop()
            
#             with st.spinner('Making prediction...'):
#                 try:
#                     # Get predictions
#                     probabilities = rf_model.predict_proba([features])[0]
#                     predicted_idx = np.argmax(probabilities)
#                     predicted_class = class_names[predicted_idx]
#                     confidence = probabilities[predicted_idx] * 100
                    
#                     # Display diagnosis
#                     if predicted_class == 'Healthy':
#                         st.success("🎉 **Diagnosis: HEALTHY**")
#                         st.success(f"**Confidence:** {confidence:.1f}%")
#                         st.info("""
#                         **Recommendations:**
#                         - Continue regular monitoring
#                         - Maintain good farming practices
#                         - Watch for early symptoms
#                         """)
#                     else:
#                         st.error(f"⚠️ **Diagnosis: {predicted_class.upper()}**")
#                         st.warning(f"**Confidence:** {confidence:.1f}%")
                        
#                         # Disease-specific recommendations
#                         with st.expander("💡 **Treatment Recommendations**"):
#                             if 'Common_Rust' in predicted_class:
#                                 st.write("""
#                                 **Common Rust Treatment:**
#                                 - Apply fungicides containing triazoles
#                                 - Remove and destroy infected plant debris
#                                 - Plant resistant varieties next season
#                                 - Avoid overhead irrigation
#                                 """)
#                             elif 'Gray_Leaf_Spot' in predicted_class:
#                                 st.write("""
#                                 **Gray Leaf Spot Treatment:**
#                                 - Use azoxystrobin-based fungicides  
#                                 - Practice crop rotation with non-host plants
#                                 - Ensure proper plant spacing for air circulation
#                                 - Remove infected crop residue after harvest
#                                 """)
#                             elif 'Northern_Leaf_Blight' in predicted_class:
#                                 st.write("""
#                                 **Northern Leaf Blight Treatment:**
#                                 - Apply chlorothalonil or mancozeb fungicides
#                                 - Increase plant spacing to reduce humidity
#                                 - Use tillage to bury crop residue
#                                 - Consider biological control agents
#                                 """)
                    
#                     # Probability distribution
#                     st.subheader("📊 Probability Distribution")
                    
#                     # Create dataframe
#                     prob_df = pd.DataFrame({
#                         'Disease': class_names,
#                         'Probability': probabilities,
#                         'Percentage': [f"{p:.1%}" for p in probabilities]
#                     }).sort_values('Probability', ascending=False)
                    
#                     # Display table
#                     st.dataframe(
#                         prob_df[['Disease', 'Percentage']],
#                         use_container_width=True,
#                         hide_index=True
#                     )
                    
#                     # Visual chart
#                     fig, ax = plt.subplots(figsize=(10, 5))
#                     colors = ['#2E86AB' if cls == predicted_class else '#A23B72' for cls in class_names]
#                     bars = ax.bar(class_names, probabilities, color=colors, alpha=0.8)
                    
#                     ax.set_title('Disease Probability Distribution', fontweight='bold', pad=20)
#                     ax.set_ylabel('Probability', fontweight='bold')
#                     ax.set_ylim(0, 1)
#                     plt.xticks(rotation=45, ha='right')
                    
#                     # Add value labels
#                     for bar, prob in zip(bars, probabilities):
#                         height = bar.get_height()
#                         ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
#                                 f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
                    
#                     plt.tight_layout()
#                     st.pyplot(fig)
                    
#                 except Exception as e:
#                     st.error(f"❌ Prediction error: {str(e)}")
                    
#     except Exception as e:
#         st.error(f"❌ Error processing image: {str(e)}")
#         st.info("Please ensure the image is valid and not corrupted")

# # ===== FOOTER & INFO =====
# st.markdown("---")
# footer_col1, footer_col2, footer_col3 = st.columns(3)

# with footer_col1:
#     st.caption("**Model Accuracy:** 92%+")
# with footer_col2:
#     st.caption("**Feature Dimensions:** 2048")
# with footer_col3:
#     st.caption("**Framework:** TensorFlow + Scikit-learn")

# # Debug info
# with st.expander("🔧 Debug Information"):
#     st.write("**Loaded Models:**")
#     st.json({
#         "Random Forest Features": getattr(rf_model, 'n_features_in_', 'Unknown'),
#         "Number of Classes": len(class_names) if class_names else 'Unknown',
#         "Class Names": class_names if class_names else 'Unknown'
#     })





# # app.py
# import os
# import streamlit as st
# import numpy as np
# from PIL import Image
# import joblib
# import pickle
# import pandas as pd
# import matplotlib.pyplot as plt
# import cv2

# # Set page config
# st.set_page_config(
#     page_title="Maize Disease Detector",
#     layout="wide",
#     page_icon="🌽"
# )

# st.title("🌽 Maize Leaf Disease Detection")
# st.markdown("Upload an image of a maize leaf for automated disease diagnosis")

# # Load models with robust error handling
# @st.cache_resource
# def load_models():
#     """Load models with comprehensive error handling"""
#     try:
#         # Check for required files
#         required_files = ['maize_rf_model.pkl', 'class_names.pkl']
#         missing_files = []
        
#         for file in required_files:
#             if not os.path.exists(file):
#                 missing_files.append(file)
        
#         if missing_files:
#             return None, None, f"Missing files: {', '.join(missing_files)}"
        
#         # Load Random Forest model
#         rf_model = joblib.load('maize_rf_model.pkl')
        
#         # Load class names
#         with open('class_names.pkl', 'rb') as f:
#             class_names = pickle.load(f)
        
#         return rf_model, class_names, None
        
#     except Exception as e:
#         return None, None, f"Error loading models: {str(e)}"

# # Feature extraction using OpenCV (no TensorFlow)
# def extract_features_opencv(img):
#     """Extract features using OpenCV instead of TensorFlow"""
#     try:
#         # Convert PIL Image to OpenCV format
#         img_cv = np.array(img)
#         img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        
#         # Resize
#         img_cv = cv2.resize(img_cv, (224, 224))
        
#         features = []
        
#         # Color features (HSV color space)
#         hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
#         for i in range(3):  # H, S, V channels
#             channel = hsv[:, :, i]
#             features.extend([
#                 np.mean(channel),
#                 np.std(channel),
#                 np.median(channel),
#                 np.min(channel),
#                 np.max(channel)
#             ])
        
#         # Texture features (grayscale)
#         gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
#         features.extend([
#             np.mean(gray),
#             np.std(gray),
#             np.median(gray)
#         ])
        
#         # Edge features
#         edges = cv2.Canny(gray, 100, 200)
#         features.extend([
#             np.mean(edges),
#             np.sum(edges > 0) / edges.size  # Edge density
#         ])
        
#         # Color histogram features (RGB)
#         for i in range(3):  # B, G, R channels
#             hist = cv2.calcHist([img_cv], [i], None, [8], [0, 256])
#             features.extend(hist.flatten())
        
#         return np.array(features).flatten()
        
#     except Exception as e:
#         st.error(f"Feature extraction error: {e}")
#         return None

# # Load models
# rf_model, class_names, error_msg = load_models()

# if error_msg:
#     st.error(f"❌ {error_msg}")
#     st.info("""
#     **Please ensure these files are in your repository:**
#     - `maize_rf_model.pkl` (Random Forest model)
#     - `class_names.pkl` (Class labels)
#     """)
#     st.stop()

# # Display success message
# st.success("✅ All models loaded successfully!")
# st.sidebar.success("System ready for analysis")

# # Show system info in sidebar
# with st.sidebar:
#     st.header("⚙️ System Information")
#     st.code(f"Python: {np.__version__}")
#     st.code(f"OpenCV: {cv2.__version__}")
#     st.code(f"NumPy: {np.__version__}")
    
#     st.header("📁 Loaded Files")
#     for file in ['maize_rf_model.pkl', 'class_names.pkl']:
#         if os.path.exists(file):
#             file_size = os.path.getsize(file) / 1024  # KB
#             st.success(f"✅ {file} ({file_size:.1f} KB)")

# # File uploader
# uploaded_file = st.file_uploader(
#     "📤 Choose a maize leaf image...", 
#     type=["jpg", "jpeg", "png"],
#     help="Select a clear, well-lit image of a maize leaf"
# )

# if uploaded_file is not None:
#     try:
#         # Load and display image
#         img = Image.open(uploaded_file).convert('RGB')
        
#         col1, col2 = st.columns([1, 1])
        
#         with col1:
#             st.image(img, caption="Uploaded Image", use_column_width=True)
#             st.caption(f"Dimensions: {img.size[0]}×{img.size[1]} pixels")
        
#         with col2:
#             st.subheader("🔍 Analysis Results")
            
#             with st.spinner('Extracting features...'):
#                 features = extract_features_opencv(img)
                
#                 if features is None:
#                     st.error("Feature extraction failed")
#                     st.stop()
                
#                 st.success(f"✓ Extracted {len(features)} features")
            
#             with st.spinner('Making prediction...'):
#                 try:
#                     # Get prediction
#                     probabilities = rf_model.predict_proba([features])[0]
#                     predicted_idx = np.argmax(probabilities)
#                     predicted_class = class_names[predicted_idx]
#                     confidence = probabilities[predicted_idx] * 100
                    
#                     # Display results
#                     if predicted_class == 'Healthy':
#                         st.success(f"🎉 **Diagnosis: HEALTHY**")
#                         st.success(f"**Confidence:** {confidence:.1f}%")
#                     else:
#                         st.error(f"⚠️ **Diagnosis: {predicted_class.upper()}**")
#                         st.warning(f"**Confidence:** {confidence:.1f}%")
                    
#                     # Probability distribution
#                     st.subheader("📊 Probability Distribution")
                    
#                     prob_data = []
#                     for cls, prob in zip(class_names, probabilities):
#                         prob_data.append({
#                             'Disease': cls,
#                             'Probability': f"{prob:.1%}",
#                             'Value': prob
#                         })
                    
#                     prob_df = pd.DataFrame(prob_data)
#                     st.dataframe(
#                         prob_df[['Disease', 'Probability']],
#                         hide_index=True,
#                         use_container_width=True
#                     )
                    
#                     # Visual chart
#                     fig, ax = plt.subplots(figsize=(10, 4))
#                     colors = ['#4CAF50' if cls == 'Healthy' else '#F44336' for cls in class_names]
#                     bars = ax.bar(class_names, probabilities, color=colors, alpha=0.8)
                    
#                     ax.set_title('Disease Probability Distribution', fontweight='bold')
#                     ax.set_ylabel('Probability')
#                     ax.set_ylim(0, 1)
#                     plt.xticks(rotation=45)
                    
#                     for bar, prob in zip(bars, probabilities):
#                         ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
#                                f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
                    
#                     st.pyplot(fig)
                    
#                 except Exception as e:
#                     st.error(f"❌ Prediction error: {str(e)}")
                    
#     except Exception as e:
#         st.error(f"❌ Error processing image: {str(e)}")

# # Footer
# st.markdown("---")
# st.caption("🌽 Maize Disease Detection | No TensorFlow | OpenCV Feature Extraction")

# app.py
import os
import sys

# CRITICAL: Force CPU and suppress TensorFlow logs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Import streamlit first to avoid conflicts
import streamlit as st
import numpy as np
from PIL import Image
import joblib
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Now import TensorFlow with error handling
try:
    import tensorflow as tf
    # Force CPU usage
    tf.config.set_visible_devices([], 'GPU')
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.preprocessing import image
    TENSORFLOW_AVAILABLE = True
    TF_VERSION = tf.__version__
except ImportError as e:
    st.error(f"TensorFlow import failed: {e}")
    TENSORFLOW_AVAILABLE = False
    TF_VERSION = "Not available"
except Exception as e:
    st.error(f"TensorFlow initialization error: {e}")
    TENSORFLOW_AVAILABLE = False
    TF_VERSION = "Error"

# Set page config
st.set_page_config(
    page_title="Maize Disease Detector",
    layout="wide",
    page_icon="🌽"
)

st.title("🌽 Maize Leaf Disease Detection")
st.markdown("Upload an image of a maize leaf for automated disease diagnosis")

@st.cache_resource
def load_models():
    """Load models with comprehensive error handling"""
    try:
        # Check for required files
        required_files = ['maize_rf_model.pkl', 'feature_extractor.h5', 'class_names.pkl']
        missing_files = []
        
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            return None, None, None, f"Missing files: {', '.join(missing_files)}"
        
        # Load Random Forest model
        rf_model = joblib.load('maize_rf_model.pkl')
        
        # Load TensorFlow feature extractor
        feature_extractor = None
        if TENSORFLOW_AVAILABLE:
            feature_extractor = tf.keras.models.load_model(
                'feature_extractor.h5', 
                compile=False
            )
        
        # Load class names
        with open('class_names.pkl', 'rb') as f:
            class_names = pickle.load(f)
        
        return rf_model, feature_extractor, class_names, None
        
    except Exception as e:
        return None, None, None, f"Error loading models: {str(e)}"

def extract_cnn_features(img, model):
    """Extract features using CNN model"""
    try:
        img = img.resize((224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x, verbose=0)
        return features.flatten()
    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        return None

# Load models
rf_model, feature_extractor, class_names, error_msg = load_models()

# Display system info in sidebar
with st.sidebar:
    st.header("⚙️ System Information")
    st.code(f"Python: {sys.version.split()[0]}")
    st.code(f"NumPy: {np.__version__}")
    st.code(f"TensorFlow: {TF_VERSION}")
    
    st.header("📁 File Status")
    for file in ['maize_rf_model.pkl', 'feature_extractor.h5', 'class_names.pkl']:
        if os.path.exists(file):
            file_size = os.path.getsize(file) / 1024 / 1024
            st.success(f"✅ {file} ({file_size:.1f} MB)")
        else:
            st.error(f"❌ {file} (Missing)")

if error_msg:
    st.error(f"❌ {error_msg}")
    st.info("""
    **Please ensure these files are in your repository:**
    - `maize_rf_model.pkl` (Random Forest model)
    - `feature_extractor.h5` (CNN feature extractor)
    - `class_names.pkl` (Class labels)
    """)
    st.stop()

if not TENSORFLOW_AVAILABLE:
    st.error("❌ TensorFlow is not available. Please check the requirements.txt")
    st.info("The app requires TensorFlow for feature extraction.")
    st.stop()

# Main application
st.success("✅ All systems ready! Upload an image to begin analysis.")

uploaded_file = st.file_uploader(
    "📤 Choose a maize leaf image...", 
    type=["jpg", "jpeg", "png"],
    help="Select a clear image of a maize leaf"
)

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(img, caption="Uploaded Image", use_column_width=True)
            st.caption(f"Dimensions: {img.size[0]}×{img.size[1]} pixels")
        
        with col2:
            st.subheader("🔍 Analysis Results")
            
            with st.spinner('Extracting CNN features...'):
                features = extract_cnn_features(img, feature_extractor)
                
                if features is None:
                    st.error("Feature extraction failed")
                    st.stop()
                
                # Verify feature dimensions
                expected_features = getattr(rf_model, 'n_features_in_', len(features))
                if len(features) != expected_features:
                    st.error(f"Feature dimension mismatch: Expected {expected_features}, got {len(features)}")
                    st.stop()
                
                st.success(f"✓ Extracted {len(features)} features")
            
            with st.spinner('Making prediction...'):
                probabilities = rf_model.predict_proba([features])[0]
                predicted_idx = np.argmax(probabilities)
                predicted_class = class_names[predicted_idx]
                confidence = probabilities[predicted_idx] * 100
                
                # Display results
                if predicted_class == 'Healthy':
                    st.success(f"🎉 **Diagnosis: HEALTHY**")
                    st.success(f"**Confidence:** {confidence:.1f}%")
                else:
                    st.error(f"⚠️ **Diagnosis: {predicted_class.upper()}**")
                    st.warning(f"**Confidence:** {confidence:.1f}%")
                
                # Probability distribution
                st.subheader("📊 Probability Distribution")
                
                prob_data = []
                for cls, prob in zip(class_names, probabilities):
                    prob_data.append({
                        'Disease': cls,
                        'Probability': f"{prob:.1%}",
                        'Value': prob
                    })
                
                prob_df = pd.DataFrame(prob_data)
                st.dataframe(
                    prob_df[['Disease', 'Probability']],
                    hide_index=True,
                    use_container_width=True
                )
                
                # Visual chart
                fig, ax = plt.subplots(figsize=(10, 4))
                colors = ['#4CAF50' if cls == 'Healthy' else '#F44336' for cls in class_names]
                bars = ax.bar(class_names, probabilities, color=colors, alpha=0.8)
                
                ax.set_title('Disease Probability Distribution', fontweight='bold')
                ax.set_ylabel('Probability')
                ax.set_ylim(0, 1)
                plt.xticks(rotation=45)
                
                for bar, prob in zip(bars, probabilities):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
                
                st.pyplot(fig)
                
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")

# Footer
st.markdown("---")
st.caption("🌽 Maize Disease Detection | TensorFlow CNN Features | Streamlit Cloud")
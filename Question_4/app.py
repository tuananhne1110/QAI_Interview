from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
from model import LogisticRegressionAPI
from utils import allowed_file, preprocess_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('Question_4', 'static', 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Load the pre-trained model
model = LogisticRegressionAPI("Question_4\logistic_regression_weights.h5")

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # Save the uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(f"Saving file to: {filepath}")  # Debugging statement
            file.save(filepath)
            
            # Preprocess the image and make a prediction
            img_data = preprocess_image(filepath)
            prediction = model.classify(img_data)
            
            # Render the result template with the prediction
            return render_template('result.html', filename=filename, prediction=prediction[0][0])
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    # Ensure the upload directory exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)

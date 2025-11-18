<div class="Header">
<h1>MidTerm Project Machine Learning Zoomcap Cohort-2025</h1>
<img src='/images/img1.jpg'>
<hr>
<br>
<h1>🔍 Shopping Behaviour</h1>
<p>This project focuses on predicting shopping behavior using machine learning models. The project includes the next steps
<ul>
<li>Data Cleaning</li>
<li>EDA (Exploratory Data Analysis)</li>
<li>Model Selection</li>
<li>Parameter tuning</li>
<li>Deployment via web service</li>
</ul>
The solution is designed for effective containerization and deployment via Docker.
<hr>
</div>
<h1>🧩 Problem Description</h1>
<hr>
<p> This dataset provides detailed insights into consumer behaviour and shopping patterns across various demographics, locations, and product categories
</p>
<br>
<h2>✅ Objective</h2>
<ul> <li>This project aims to develop a machine learning model that explore <b>consumer decision-making</b> and <b>market trends</b></li>
Also,
<li>Analyzing How Customer and Product Variables Influence Review Ratings</li>
</ul>
<br>
<h2>📊 Dataset</h2>
<p>This project uses an dataset provides by Kaggle</p>
<b>Link:</b> <a href="https://www.kaggle.com/datasets/zubairamuti/shopping-behaviours-dataset">Dataset link</a>
<p>The dataset contains around 3,900 customer records with 18 attributes that describe purchase details, shopping habits, and preferences</p>
<p>it includes information such as: </p>
<ul>
    <li><b>Customer demographics:</b> age, gender or location</li>
    <li><b>Product details: </b> item purchased, category, size, color, season</li>
    <li><b>Purchase information: </b> amount spent in USD, payment method, shipping type</li>
    <li><b>Shopping behaviour:</b>frequency of purchases, previous purchases, subscription status, discount usage, promo codes</li>
    <li><b>Customer feedback: </b> review ratings</li>
</ul>
<p>For more details information about the dataset, including explanations of each columns, please refear to the like share, or, <a href="https://www.kaggle.com/datasets/zubairamuti/shopping-behaviours-dataset">Click here</a>
Also, dataset name is <code>shopping_behaviour_updated.csv</code>
<br>
<h2>🛠️ Notebook</h2>
<hr>
Notebook's section includes a file named <code>notebook.ipynb</code>
<hr>
<h2>🧠 Model Training</h2>
<p>The training script evaluates the following models:</p>
<ul>
  <li>Linear Regression</li>
  <li>Random Forest Regressor</li>
  <li>XGBoost Regressor</li>
  <li>Gradient Boosting Regressor</li>
</ul>
<p>The model with the lowest RMSE on the validation set is saved as the final artifact for deployment.</p>
<h2>🧷 Scripts</h2>
<hr>
Scripts section has 2 files named <code>train.py</code> and <code>predict.py</code>
<hr>

<h2>🌐 Flask Web Service</h2>
<p>The API exposes two main routes:</p>

<h4>🔹 <code>GET /</code></h4>
<p>Health check endpoint returning a basic running message.</p>

<h4>🔹 <code>POST /predict</code></h4>
<p>Accepts a JSON payload and returns the predicted rating.</p>

<p><strong>Example Request:</strong></p>
<pre><code>curl -X POST http://localhost:9696/predict \
     -H "Content-Type: application/json" \
     -d '{"category":"Electronics","gender":"M","purchase_amount":200}'

</code></pre>


<h2>🛠️ requirements.txt</h2>
<p>The project uses the following Python dependencies:</p>
<pre><code>flask
pandas
numpy
scikit-learn
xgboost
gunicorn
</code></pre>
<p>Add any additional dependencies from your notebook if necessary.</p>

<h2>🧷 Dockerfile</h2>
<hr>
A <code>Dockerfile</code> is provided to build and run the service in <code>Docker container</code>
<h2>🐳 Docker Deployment</h2>
<h3>📦 Build the Docker Image</h3>
<code>docker build -t midterm-service .</code>
<h2>▶️ Run the Docker Container</h2>
<code>docker run -p 9696:9696 midterm-service</code>



<h2>▶️ Full Local Workflow</h2>
<ol>
  <li>Explore and clean the data using the notebook.</li>
  <li>Train ML models with <code>train.py</code>.</li>
  <li>Save the trained model and <code>DictVectorizer</code> inside <code>model/</code>.</li>
  <li>Run the prediction service locally with <code>Flask</code>.</li>
  <li>Test predictions using <code>curl</code> or <code>Postman</code>.</li>
  <li>Package and deploy the service using <code>Docker</code>.</li>
</ol>

<h2>✔️ Example of JSON prediction</h2>
<code>
{
  "category": "Electronics",
  "shipping_type": "express",
  "gender": "M",
  "purchase_amount": 150,
  "age": 32
}
</code>

<h2>✔️ Notes</h2>
<ul>
  <li>The preprocessing logic mirrors the notebook implementation.</li>
  <li>Model artifacts must be generated prior to executing the Docker container.</li>
  <li>If the dataset is modified, retraining the model is required.</li>
</ul>

<b>Elaborated by Iver Samuel Medina Balboa - ML Zoomcap Cohort 2025 - IA Student from Computer Science UMSA</b>
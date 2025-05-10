import pickle
import numpy as np
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login
from django.contrib.auth.forms import UserCreationForm
from sklearn.metrics import accuracy_score, log_loss
from django.contrib.auth import logout
import base64
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from io import BytesIO

# Load the trained model from pickle
with open('../model/employee_attrition_model.pkl', 'rb') as file:
    model = pickle.load(file)


# Home Page View
def home(request):
    return render(request, 'home.html')  # ✅ Fixed template path

# Login View
def login_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect("predictionForm")
        else:
            return JsonResponse({"error": "Invalid username or password"}, status=400)

    return render(request, "login.html")

# Register View
def register_view(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)  # Auto-login after registration
            return redirect("predictionForm")
    else:
        form = UserCreationForm()

    return render(request, "register.html", {"form": form})



def predictionForm(request):
    # Render the prediction form
    return render(request, 'predictionForm.html')



@login_required
# Prediction page (Displays result)
def predictionResult(request):
    if request.method == "POST":
        try:
            # Collect form data from POST request
            age = int(request.POST.get('Age'))
            gender = int(request.POST.get('Gender'))
            years_at_company = int(request.POST.get('Years at Company'))
            monthly_income = int(request.POST.get('Monthly Income'))
            work_life_balance = int(request.POST.get('Work-Life Balance'))
            job_satisfaction = int(request.POST.get('Job Satisfaction'))
            performance_rating = int(request.POST.get('Performance Rating'))
            num_promotions = int(request.POST.get('Number of Promotions'))
            overtime = int(request.POST.get('Overtime'))
            distance_from_home = float(request.POST.get('Distance from Home'))
            education_level = int(request.POST.get('Education Level'))
            marital_status = int(request.POST.get('Marital Status'))
            num_dependents = int(request.POST.get('Number of Dependents'))
            job_level = int(request.POST.get('Job Level'))
            company_size = int(request.POST.get('Company Size'))
            company_tenure = float(request.POST.get('Company Tenure (In Months)'))
            remote_work = int(request.POST.get('Remote Work'))
            company_reputation = int(request.POST.get('Company Reputation'))
            employee_recognition = int(request.POST.get('Employee Recognition'))

            # Prepare input features for prediction
            input_data = np.array([[
                age, gender, years_at_company, monthly_income, work_life_balance, job_satisfaction,
                performance_rating, num_promotions, overtime, distance_from_home, education_level, marital_status,
                num_dependents, job_level, company_size, company_tenure, remote_work, company_reputation, employee_recognition
            ]])

            # Make prediction
            prediction = model.predict(input_data)

            # Convert the prediction (0 or 1) to a result
            if prediction[0] == 1:
                result = "Employee will leave"
            else:
                result = "Employee will stay"

            

            return render(request, 'predictionResult.html', {'result': result})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

    return JsonResponse({'error': 'Invalid request'}, status=400)





@login_required
def dashboard(request):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from io import BytesIO
    import base64
    import pandas as pd
    import pickle
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, log_loss
    from xgboost import XGBClassifier  # ✅ Ensure XGBoost is used for feature importance

    # Load dataset
    df = pd.read_csv('../model/F_Employee_Attrition_Prediction.csv')  # ✅ Fixed path

    # Ensure 'Attrition' is excluded from features
    if 'Attrition' in df.columns:
        X = df.drop('Attrition', axis=1)  # ✅ Exclude 'Attrition' column
        y = df['Attrition']  # ✅ Use 'Attrition' as the target variable
    else:
        raise ValueError("The dataset does not contain the 'Attrition' column.")

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load model
    with open('../model/employee_attrition_model.pkl', 'rb') as f:  # ✅ Fixed path
        model = pickle.load(f)

    # Ensure the columns match the training data
    expected_columns = ['Age', 'Gender', 'Years at Company', 'Monthly Income', 'Work-Life Balance',
                        'Job Satisfaction', 'Performance Rating', 'Number of Promotions', 'Overtime',
                        'Distance from Home', 'Education Level', 'Marital Status', 'Number of Dependents',
                        'Job Level', 'Company Size', 'Company Tenure (In Months)', 'Remote Work',
                        'Company Reputation', 'Employee Recognition']
    X_test = X_test[expected_columns]

    # Predict on test set
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Compute accuracy and log loss
    accuracy = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_proba)

    # Attrition distribution data
    target_counts = df['Attrition'].value_counts(normalize=True) * 100
    target_percent = {k: round(v, 2) for k, v in target_counts.items()}

    # Bar Chart (Attrition Distribution)
    plt.figure(figsize=(5, 4))
    sns.countplot(x='Attrition', data=df, palette='Set2')
    plt.title('Employee Attrition Distribution')
    plt.xlabel('Attrition Status (Yes / No)')
    plt.ylabel('Number of Employees')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    bar_chart = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    # **Feature Importance Bar Chart**
    feature_importances = model.feature_importances_
    feature_names = X.columns

    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False).head(19)  # ✅ Top 10 features

    plt.figure(figsize=(8, 5))
    sns.barplot(y=importance_df['Feature'], x=importance_df['Importance'], palette='Blues_r')
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Feature Name')
    plt.title('Top Factors Affecting Employee Attrition')
    plt.figtext(0.1, -0.05, "🔹 Higher importance means the feature has a stronger impact on attrition", fontsize=10, color="black")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    feature_chart = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    # Render template
    context = {
        'total_records': len(df),
        'model_accuracy': f"{accuracy * 100:.2f}%",  # ✅ Accuracy from test set
        'loss_value': f"{loss:.4f}",  # ✅ Log loss from test set
        'target_percent': target_percent,
        'bar_chart': bar_chart,  # ✅ Attrition Distribution Chart
        'feature_chart': feature_chart  # ✅ Feature Importance Chart
    }
    return render(request, 'dashboard.html', context)


def logout_view(request):
    logout(request)  # Logs out the user
    return redirect('login')  # Redirect to the login page after logout
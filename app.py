from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import json
import os


from quiz import quiz_questions

app = Flask(__name__)

app.secret_key = os.getenv("SECRET_KEY")


# Load model and vectorizer
model = joblib.load('model/mbti_model.pkl')
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')

# Load metadata
with open('data/mbti_traits.json') as f:
    mbti_traits = json.load(f)

with open('data/mbti_rarity.json') as f:
    mbti_rarity = json.load(f)

with open('data/mbti_celebrities.json') as f:
    mbti_celebrities = json.load(f)

with open('data/mbti_funparagraph.json') as f:
    mbti_fun_lines = json.load(f)



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        user_intro = request.form['user_intro']
        if user_intro.strip() == "":
            return render_template('chat.html', error="Please write something!")

        # Predict MBTI from text
        X = vectorizer.transform([user_intro])
        mbti_pred = model.predict(X)[0]
        session['mbti'] = mbti_pred
        return redirect(url_for('loading'))

    return render_template('chat.html')


@app.route('/quiz', methods=['GET', 'POST'])
def quiz():
    if 'quiz_index' not in session:
        session['quiz_index'] = 0
        session['answers'] = []

    index = session['quiz_index']

    if request.method == 'POST':
        answer = request.form.get('option')
        if answer:
            session['answers'].append(answer)
            session['quiz_index'] += 1
            index += 1

    if index < len(quiz_questions):
        question = quiz_questions[index]
        return render_template('quiz.html', question=question, index=index + 1, total=len(quiz_questions))

    # Calculate MBTI from answers
    answers = session.pop('answers', [])
    session.pop('quiz_index', None)
    mbti = ""

    for trait1, trait2 in [('E', 'I'), ('S', 'N'), ('T', 'F'), ('J', 'P')]:
        count1 = answers.count(trait1)
        count2 = answers.count(trait2)
        mbti += trait1 if count1 >= count2 else trait2

    session['mbti'] = mbti
    return redirect(url_for('loading'))


@app.route('/loading')
def loading():
    # Optional: fallback redirect if no MBTI in session
    if 'mbti' not in session:
        return redirect(url_for('index'))
    return render_template('loading.html')


@app.route('/result')
def result():
    mbti = session.get('mbti')
    if not mbti:
        return redirect(url_for('index'))

    traits = mbti_traits.get(mbti, "No trait data available.")
    rarity = mbti_rarity.get(mbti, "Unknown")
    famous_people = mbti_celebrities.get(mbti, [])

    fun_paragraph = mbti_fun_lines.get(mbti, f"You are a {mbti}. Apparently, {rarity.lower()} personality types do exist.")

    return render_template('result.html',
                           mbti=mbti,
                           traits=traits,
                           rarity=rarity,
                           famous_people=famous_people,
                           fun_paragraph=fun_paragraph)


@app.route('/related_apps')
def related_apps():
    return render_template('related_apps.html')


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/how-it-works')
def how_it_works():
    return render_template('how_it_works.html')
def get_rarity_data():
    with open(os.path.join('data', 'mbti_rarity.json'), 'r') as f:
        return json.load(f)



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  
    app.run(host='0.0.0.0', port=port)

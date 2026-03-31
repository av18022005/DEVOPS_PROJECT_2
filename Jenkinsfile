pipeline {
    agent any

    stages {

        stage('Install Dependencies') {
            steps {
                bat 'python -m venv venv'
                bat 'venv\\Scripts\\activate && pip install -r requirements.txt'
            }
        }

        stage('Run CI Pipeline') {
            steps {
                bat 'venv\\Scripts\\activate && python ci_pipeline.py'
            }
        }
    }
}
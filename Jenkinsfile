pipeline {
    agent any

    stages {

        stage('Clone Repository') {
            steps {
                git 'https://github.com/av18022005/DEVOPS_PROJECT.git'
            }
        }

        stage('Setup Virtual Environment') {
            steps {
                bat 'python -m venv venv'
                bat 'venv\\Scripts\\activate && pip install --upgrade pip'
                bat 'venv\\Scripts\\activate && pip install -r requirements.txt'
            }
        }

        stage('Install Torch') {
            steps {
                bat 'venv\\Scripts\\activate && pip install torch torchvision'
            }
        }

        stage('Run ML Pipeline') {
            steps {
                bat 'venv\\Scripts\\activate && python main.py'
            }
        }

        stage('Check Decision') {
            steps {
                script {
                    if (fileExists('logs/decision.txt')) {
                        def decision = readFile('logs/decision.txt').trim()
                        echo "Decision: ${decision}"

                        if (decision == "REJECT") {
                            error("❌ Model Rejected - Pipeline Stopped")
                        }
                    } else {
                        echo "No decision file found"
                    }
                }
            }
        }
    }

    post {
        success {
            echo '✅ Model Approved & Pipeline Successful'
        }
        failure {
            echo '❌ Pipeline Failed'
        }
    }
}
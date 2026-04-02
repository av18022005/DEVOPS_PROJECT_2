pipeline {
    agent any

    options {
        timeout(time: 40, unit: 'MINUTES')   // ⏱️ total pipeline timeout
    }

    environment {
        VENV = "venv"
        PYTHON = "venv\\Scripts\\python.exe"
        PIP = "venv\\Scripts\\pip.exe"
    }

    stages {

        stage('Install Dependencies') {
            steps {
                script {
                    // Create virtual environment if not exists
                    if (!fileExists(env.VENV)) {
                        bat "python -m venv ${env.VENV}"
                    }

                    bat "${env.PIP} install --upgrade pip"
                    bat "${env.PIP} install -r requirements.txt"
                }
            }
        }

        stage('Prepare Directories') {
            steps {
                // Ensure required folders exist (prevents permission errors)
                bat 'if not exist reports mkdir reports'
                bat 'if not exist models mkdir models'
            }
        }

        stage('Run CI Pipeline') {
            steps {
                bat "${env.PYTHON} ci_pipeline.py"
            }
        }

        stage('Archive Results') {
            steps {
                archiveArtifacts artifacts: 'reports/*.json', fingerprint: true
            }
        }
    }

    post {
        success {
            echo '✅ Build Successful: Model Passed CI Checks'
        }
        failure {
            echo '❌ Build Failed: Model Rejected or Error Occurred'
        }
    }
}
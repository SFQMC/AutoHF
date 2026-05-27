properties([
  disableConcurrentBuilds(),
  buildDiscarder(logRotator(numToKeepStr: '8', daysToKeepStr: '20'))
])

timeout(time: 1, unit: 'HOURS') {
  buildPod(context: 'docker', dockerfile: 'Dockerfile') {
    withEnv([
      "SRC=$WORKSPACE",
      "VENV=/opt/venv",
    ]) {
      stage('pip install') {
        sh '''
          . $VENV/bin/activate
          cd $SRC
          pip install --no-deps .[TESTING,DOCS,cuda13]
        '''
      }
      parallel(
        'tests': {
          stage('pytest') {
            sh '''
              . $VENV/bin/activate
              cd $SRC
              pytest
            '''
          }
        },
        'docs': {
          stage('build docs') {
            sh '''
              . $VENV/bin/activate
              cd $SRC/docs
              make html
            '''
          }
          if (env.BRANCH_NAME in ['main', 'ci'] || env.TAG_NAME) {
            stage('deploy docs') {
              def scm = scmGit(branches: [[name: 'refs/heads/gh-pages']],
                userRemoteConfigs: [[credentialsId: 'github-jenkins', url: 'https://github.com/SFQMC/sfqmc.github.io.git']])
              dir(path: 'sfqmc.github.io') {
                checkout(changelog: false, poll: false, scm: scm)
                sh '''#!/bin/bash -e
                  rm -rf autohf/$BRANCH_NAME
                  mkdir -p autohf/$BRANCH_NAME
                  cp -a $SRC/docs/_build/html/. autohf/$BRANCH_NAME
                  if [ -n "$TAG_NAME" ]; then
                    LATEST=$(git -C $SRC tag --sort=-v:refname | head -n1)
                    ln -sfn "$LATEST" autohf/stable
                  fi
                  git add -A autohf
                  if git diff --cached --quiet; then
                    echo "No documentation changes to deploy."
                  else
                    GIT_COMMITTER_EMAIL="jenkins@flatironinstitute.org" GIT_COMMITTER_NAME="AutoHF CI" git commit --author='AutoHF CI <jenkins@flatironinstitute.org>' --allow-empty -m "Update AutoHF docs from build $BUILD_NUMBER ($GIT_COMMIT)"
                  fi
                '''
                gitPush(gitScm: scm, targetBranch: 'gh-pages', targetRepo: 'origin')
              }
            }
          }
        }
      )
    }
  }
}

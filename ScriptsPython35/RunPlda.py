import os
import subprocess


def generateTopics(num_topics):
    for i in range(1, 6):
        args = 'plda\\lda.exe --num_topics ' + str(num_topics) + ' '
        args += '--alpha 0.1 --beta 0.001 --training_data_file plda\\lda-format-files\\treino' + str(i) + '.txt '
        args += '--model_file plda\\topic' + str(num_topics) + '\\treino' + str(i) + '_lda_model.txt '
        args += '--burn_in_iterations 100 --total_iterations 150'
        subprocess.call(args)

def inferenceDocs(num_topics):
    for i in range(1,6):
        args = 'plda\\infer.exe --alpha 0.1 --beta 0.01 --inference_data_file plda\\lda-format-files\\teste'+str(i)+'.txt '
        args += '--inference_result_file plda\\topic'+str(num_topics)+'\\treino'+str(i)+'_inference_result.txt '
        args += '--model_file plda\\topic'+str(num_topics)+'\\treino'+str(i)+'_lda_model.txt '
        args += '--total_iterations 15 --burn_in_iterations 10'
        subprocess.call(args)
if __name__ == '__main__':
    num_topics = 100
    #generateTopics(num_topics)
    inferenceDocs(num_topics)
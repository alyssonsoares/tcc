import os
import subprocess



if __name__ == '__main__':
    num_topics = 100
    caminho = 'plda\\lda-format-files'
    for i in range(1,6):
        args = 'plda\\lda.exe --num_topics '+str(num_topics)+' '
        args += '--alpha 0.1 --beta 0.001 --training_data_file plda\\lda-format-files\\treino'+str(i)+'.txt '
        args += '--model_file plda\\topic'+str(num_topics)+'\\treino'+str(i)+'_lda_model.txt '
        args += '--burn_in_iterations 100 --total_iterations 150'
        subprocess.call(args)
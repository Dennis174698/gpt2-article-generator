import gpt_2_simple as gpt2

if __name__ == '__main__':
    file_name = "articles2.txt"

    # gpt2.copy_file_from_gdrive(file_name)

    sess = gpt2.start_tf_sess()
    if not gpt2.is_gpt2_downloaded():
        gpt2.download_gpt2()

    # DEFAULT_CONFIG = {
    #     'model_name': '124M',
    #     'run_name': '124M_article_generator_model',  # The name of the model
    #     'top_k': '0',  # How many previous words to consider when generating a new word. 0 means unlimited
    #     'include_prefix': 'True',
    #     'return_as_list': 'True',
    #     'truncate': '<|endoftext|><|startoftext|>'  # Truncate the sample where it contains this substring
    # }
    # gpt2.load_gpt2(sess, run_name='124M_article_generator_model',
    #                checkpoint_dir="checkpoint",
    #                model_dir='models'
    #                )

    gpt2.finetune(sess,
                  model_dir='models',
                  checkpoint_dir="checkpoint1",
                  dataset=file_name,
                  model_name='124M',
                  steps=1000,
                  restore_from='fresh',
                  run_name='124M_article_generator_model',
                  print_every=10,
                  )

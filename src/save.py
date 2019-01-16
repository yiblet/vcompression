import tensorflow as tf
from global_variables import *

if __name__ == "__main__":
    additional_flags = Namespace()
    additional_flags.local = False
    additional_flags.run_subprocesses = False

    def modify_parser(parser):
        parser.add_argument(
            'save_name',
            type=str,
            help='new save location',
        )

    define_flags(
        additional_flags=additional_flags,
        modify_parser=modify_parser,
    )

    if FLAGS.debug >= 1:
        pprint.pprint(FLAGS.__dict__)

    if (tf.gfile.Exists(FLAGS.summaries_dir)):
        try:
            tf.gfile.MakeDirs(FLAGS.saves_dir)
            for file, dir in [
                (f'{FLAGS.summaries_dir}/{dir}_{FLAGS.run_type}', dir)
                for dir in [FLAGS.train_dir, FLAGS.test_dir]
            ]:
                tf.gfile.Rename(
                    file, f'{FLAGS.saves_dir}/{FLAGS.save_name}_{dir}'
                )
            print('sucess!')
        except Exception as e:
            print(e)
            raise e
    else:
        print('summaries do not exist')

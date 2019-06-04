import traceback
from tensorboardX import SummaryWriter
from threading import Thread
from urllib.request import Request, urlopen
import json


class Logger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        self.train_stats = {}
        self.eval_stats = {}

    def tb_model_weights(self, model, step):
        layer_num = 1
        for name, param in model.named_parameters():
            self.writer.add_scalar(
                "layer{}-{}/max".format(layer_num, name),
                param.max(), step)
            self.writer.add_scalar(
                "layer{}-{}/min".format(layer_num, name),
                param.min(), step)
            self.writer.add_scalar(
                "layer{}-{}/mean".format(layer_num, name),
                param.mean(), step)
            self.writer.add_scalar(
                "layer{}-{}/std".format(layer_num, name),
                param.std(), step)
            self.writer.add_histogram(
                "layer{}-{}/param".format(layer_num, name), param, step)
            self.writer.add_histogram(
                "layer{}-{}/grad".format(layer_num, name), param.grad, step)
            layer_num += 1

    def dict_to_tb_scalar(self, scope_name, stats, step):
        for key, value in stats.items():
            self.writer.add_scalar('{}/{}'.format(scope_name, key), value, step)

    def dict_to_tb_figure(self, scope_name, figures, step):
        for key, value in figures.items():
            self.writer.add_figure('{}/{}'.format(scope_name, key), value, step)

    def dict_to_tb_audios(self, scope_name, audios, step, sample_rate):
        for key, value in audios.items():
            try:
                self.writer.add_audio('{}/{}'.format(scope_name, key), value, step, sample_rate=sample_rate)
            except:
                traceback.print_exc()

    def tb_train_iter_stats(self, step, stats):
        self.dict_to_tb_scalar("TrainIterStats", stats, step)
    
    def tb_train_epoch_stats(self, step, stats):
        self.dict_to_tb_scalar("TrainEpochStats", stats, step)

    def tb_train_figures(self, step, figures):
        self.dict_to_tb_figure("TrainFigures", figures, step)

    def tb_train_audios(self, step, audios, sample_rate):
        self.dict_to_tb_audios("TrainAudios", audios, step, sample_rate)

    def tb_eval_stats(self, step, stats):
        self.dict_to_tb_scalar("EvalStats", stats, step)

    def tb_eval_figures(self, step, figures):
        self.dict_to_tb_figure("EvalFigures", figures, step)

    def tb_eval_audios(self, step, audios, sample_rate):
        self.dict_to_tb_audios("EvalAudios", audios, step, sample_rate)
    
    def tb_test_audios(self, step, audios, sample_rate):
        self.dict_to_tb_audios("TestAudios", audios, step, sample_rate)

    def tb_test_figures(self, step, figures):
        self.dict_to_tb_figure("TestFigures", figures, step)


def log_to_slack(slack_url, msg=None):
    if (msg is not None) and (slack_url is not None):
        Thread(target=_send_slack, args=(slack_url, "TTS ", msg,)).start()

def _send_slack(slack_url, header, msg):
    try:
        req = Request(slack_url)
        req.add_header('Content-Type', 'application/json')
        urlopen(req, json.dumps({
            'username': 'tacotron',
            'icon_emoji': ':taco:',
            'text': '*%s*: %s' % (header, msg)
        }).encode())
    except Exception as e:
        print(e)
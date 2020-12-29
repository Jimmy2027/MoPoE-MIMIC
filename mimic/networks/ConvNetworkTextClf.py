import torch.nn as nn

from mimic.networks.char_encoding.FeatureExtractorText import make_res_block_enc_feat_ext


class ClfText(nn.Module):
    def __init__(self, flags, labels):
        super(ClfText, self).__init__()
        self.args = flags
        self.labels = labels
        if flags.text_encoding == 'char':
            self.conv1 = nn.Conv1d(self.args.num_features, self.args.DIM_text,
                                   kernel_size=4, stride=2, padding=1, dilation=1)
        elif flags.text_encoding == 'word':
            self.embedding = nn.Embedding(num_embeddings=self.args.vocab_size, embedding_dim=self.args.DIM_text,
                                          padding_idx=0)
            self.conv1 = nn.Conv1d(self.args.DIM_text, self.args.DIM_text,
                                   kernel_size=4, stride=2, padding=1, dilation=1)
        self.resblock_1 = make_res_block_enc_feat_ext(self.args.DIM_text,
                                                      2 * self.args.DIM_text,
                                                      kernelsize=4, stride=2, padding=1, dilation=1)
        self.resblock_2 = make_res_block_enc_feat_ext(2 * self.args.DIM_text,
                                                      3 * self.args.DIM_text,
                                                      kernelsize=4, stride=2, padding=1, dilation=1)
        self.resblock_3 = make_res_block_enc_feat_ext(3 * self.args.DIM_text,
                                                      4 * self.args.DIM_text,
                                                      kernelsize=4, stride=2, padding=1, dilation=1)
        self.resblock_4 = make_res_block_enc_feat_ext(4 * self.args.DIM_text,
                                                      4 * self.args.DIM_text,
                                                      kernelsize=4, stride=2, padding=1, dilation=1)
        self.resblock_5 = make_res_block_enc_feat_ext(4 * self.args.DIM_text,
                                                      4 * self.args.DIM_text,
                                                      kernelsize=4, stride=2, padding=1, dilation=1)
        self.resblock_6 = make_res_block_enc_feat_ext(4 * self.args.DIM_text,
                                                      5 * self.args.DIM_text,
                                                      kernelsize=4, stride=2, padding=1, dilation=1)
        self.resblock_7 = make_res_block_enc_feat_ext(5 * self.args.DIM_text,
                                                      5 * self.args.DIM_text,
                                                      kernelsize=4, stride=2, padding=1, dilation=1)
        self.resblock_8 = make_res_block_enc_feat_ext(5 * self.args.DIM_text,
                                                      5 * self.args.DIM_text,
                                                      kernelsize=4, stride=2, padding=0, dilation=1)

        self.dropout = nn.Dropout(p=0.5, inplace=False)
        self.linear = nn.Linear(in_features=5 * flags.DIM_text, out_features=len(self.labels), bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_text):
        """
        input_shape: [batch_size, len_sentence, vocab_size]
        Example:
            torch.Size([10, 1024, 71])
            torch.Size([10, 71, 1024])
            torch.Size([10, 128, 512])
            torch.Size([10, 256, 256])
            torch.Size([10, 384, 128])
            torch.Size([10, 512, 64])
            torch.Size([10, 512, 32])
            torch.Size([10, 512, 16])
            torch.Size([10, 640, 8])
            torch.Size([10, 640, 4])
            torch.Size([10, 640, 1])
            torch.Size([10, 640, 1])
            torch.Size([10, 640, 1])
            torch.Size([10, 3])
            torch.Size([10, 3])
        """
        if self.args.text_encoding == 'word':
            out = self.embedding(x_text.long())
            out = out.transpose(-2, -1)
            out = self.conv1(out)
        elif self.args.text_encoding == 'char':
            x_text = x_text.transpose(-2, -1)
            out = self.conv1(x_text)
        out = self.resblock_1(out)
        out = self.resblock_2(out)
        out = self.resblock_3(out)
        out = self.resblock_4(out)
        out = self.resblock_5(out)
        out = self.resblock_6(out)
        if self.args.len_sequence > 500:
            out = self.resblock_7(out)
            out = self.resblock_8(out)
        h = self.dropout(out)
        h = h.view(h.size(0), -1)
        h = self.linear(h)
        out = self.sigmoid(h)
        return out;

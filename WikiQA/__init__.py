from    argparse                    import    Namespace
from    bz2                         import    BZ2File
from    collections                 import    Counter
from    copy                        import    copy, deepcopy
from    gzip                        import    open as gopen
from    json                        import    dumps, load as jlod, loads
from    multiprocessing             import    Pool
from    multiprocessing.util        import    Finalize
from    numpy                       import    argmax, argpartition, argsort, array, float_, int_, load as nlod, log, log1p, multiply, unique, unravel_index
from    numpy.random                import    random, shuffle
from    os                          import    mkdir, remove, rename
from    os.path                     import    isdir, isfile
from    wexpect                     import    spawn
from    prettytable                 import    PrettyTable
from    re                          import    findall
from    regex                       import    compile, match, split
from    scipy.sparse                import    csr_matrix
from    shutil                      import    rmtree
from    sklearn.utils               import    murmurhash3_32
from    spacy                       import    load
from    sqlite3                     import    connect
from    tarfile                     import    open as topen
from    termcolor                   import    colored
from    time                        import    time
from    torch                       import    ByteTensor, cat, device, ger, is_tensor, load as tlod, LongTensor, ones, save, sort, Tensor, zeros
from    torch.nn                    import    DataParallel, Embedding, GRU, Linear, LSTM, Module, ModuleList, RNN
from    torch.nn.functional         import    dropout, log_softmax, nll_loss, relu, softmax
from    torch.nn.utils              import    clip_grad_norm
from    torch.nn.utils.rnn          import    pack_padded_sequence, PackedSequence, pad_packed_sequence
from    torch.optim                 import    Adamax, SGD
from    torch.utils.data            import    DataLoader, Dataset
from    torch.utils.data.sampler    import    Sampler
from    tqdm                        import    tqdm
from    unicodedata                 import    normalize
from    urllib.request              import    urlretrieve
from    zipfile                     import    ZipFile
PROCESS_TOK, PROCESS_DB, PROCESS_CANDS, Wordlist = None, None, None, ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn', "'ll", "'re", "'ve", "n't", "'s", "'d", "'m", "''", "``"]
class StackedBRNN(Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0, dropout_output=False, rnn_type=LSTM, concat_layers=False, padding=False):
        super(StackedBRNN, self).__init__()
        self.padding, self.dropout_output, self.dropout_rate, self.num_layers, self.concat_layers, self.rnns = padding, dropout_output, dropout_rate, num_layers, concat_layers, ModuleList()
        for i in range(num_layers):
            input_size = input_size if i == 0 else 2 * hidden_size
            self.rnns.append(rnn_type(input_size, hidden_size, num_layers=1, bidirectional=True))
    def forward(self, x, x_mask):
        if x_mask.data.sum() == 0: output = self._forward_unpadded(x, x_mask)
        elif self.padding or not self.training: output = self._forward_padded(x, x_mask)
        else: output = self._forward_unpadded(x, x_mask)
        return output.contiguous()
    def _forward_unpadded(self, x, x_mask):
        x = x.transpose(0, 1)
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]
            if self.dropout_rate > 0: rnn_input = dropout(rnn_input, p=self.dropout_rate,  training=self.training)
            rnn_output = self.rnns[i](rnn_input)[0]
            outputs.append(rnn_output)
        if self.concat_layers: output = cat(outputs[1:], 2)
        else: output = outputs[-1]
        output = output.transpose(0, 1)
        return dropout(output, p=self.dropout_rate, training=self.training) if self.dropout_output and self.dropout_rate > 0 else output
    def _forward_padded(self, x, x_mask):
        lengths = x_mask.data.eq(0).long().sum(1).squeeze()
        _, idx_sort = sort(lengths, dim=0, descending=True)
        _, idx_unsort = sort(idx_sort, dim=0)
        rnn_input = pack_padded_sequence(x.index_select(0, idx_sort).transpose(0, 1), list(lengths[idx_sort]))
        outputs = [rnn_input]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]
            if self.dropout_rate > 0:
                dropout_input = dropout(rnn_input.data, p=self.dropout_rate, training=self.training)
                rnn_input = PackedSequence(dropout_input, rnn_input.batch_sizes)
            outputs.append(self.rnns[i](rnn_input)[0])
        for i, o in enumerate(outputs[1:], 1): outputs[i] = pad_packed_sequence(o)[0]
        output = cat(outputs[1:], 2).transpose(0, 1).index_select(0, idx_unsort) if self.concat_layers else outputs[-1].transpose(0, 1).index_select(0, idx_unsort)
        if output.size(1) != x_mask.size(1): output = cat([output, zeros(output.size(0), x_mask.size(1) - output.size(1), output.size(2)).type(output.data.type())], 1)
        return dropout(output, p=self.dropout_rate, training=self.training) if self.dropout_output and self.dropout_rate > 0 else output
class SeqAttnMatch(Module):
    def __init__(self, input_size, identity=False):
        super(SeqAttnMatch, self).__init__()
        self.linear = Linear(input_size, input_size) if not identity else None
    def forward(self, x, y, y_mask):
        if self.linear:
            x_proj = self.linear(x.view(-1, x.size(2))).view(x.size())
            x_proj, y_proj = relu(x_proj), self.linear(y.view(-1, y.size(2))).view(y.size())
            y_proj = relu(y_proj)
        else: x_proj, y_proj = x, y
        scores = x_proj.bmm(y_proj.transpose(2, 1))
        y_mask = y_mask.unsqueeze(1).expand(scores.size())
        scores.data.masked_fill_(y_mask.data, -float('inf'))
        return softmax(scores.view(-1, y.size(1)), dim=-1).view(-1, x.size(1), y.size(1)).bmm(y)
class BilinearSeqAttn(Module):
    def __init__(self, x_size, y_size, identity=False, normalize=True):
        super(BilinearSeqAttn, self).__init__()
        self.normalize, self.linear = normalize, Linear(y_size, x_size) if not identity else None
    def forward(self, x, y, x_mask):
        Wy = self.linear(y) if self.linear is not None else y
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        if self.normalize: return log_softmax(xWy, dim=-1) if self.training else softmax(xWy, dim=-1)
        return xWy.exp()
class LinearSeqAttn(Module):
    def __init__(self, input_size):
        super(LinearSeqAttn, self).__init__()
        self.linear = Linear(input_size, 1)
    def forward(self, x, x_mask):
        scores = self.linear(x.view(-1, x.size(-1))).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        return softmax(scores, dim=-1)
class RnnDocReader(Module):
    RNN_TYPES = {'lstm': LSTM, 'gru': GRU, 'rnn': RNN}
    def __init__(self, args, normalize=True):
        super(RnnDocReader, self).__init__()
        self.args = args
        self.embedding = Embedding(args.vocab_size,args.embedding_dim,padding_idx=0)
        if args.use_qemb: self.qemb_match = SeqAttnMatch(args.embedding_dim)
        doc_input_size = args.embedding_dim + args.num_features
        if args.use_qemb: doc_input_size += args.embedding_dim
        self.doc_rnn = StackedBRNN(input_size=doc_input_size, hidden_size=args.hidden_size, num_layers=args.doc_layers, dropout_rate=args.dropout_rnn, dropout_output=args.dropout_rnn_output, concat_layers=args.concat_rnn_layers, rnn_type=self.RNN_TYPES[args.rnn_type], padding=args.rnn_padding,)
        self.question_rnn = StackedBRNN(input_size=args.embedding_dim, hidden_size=args.hidden_size, num_layers=args.question_layers, dropout_rate=args.dropout_rnn, dropout_output=args.dropout_rnn_output, concat_layers=args.concat_rnn_layers, rnn_type=self.RNN_TYPES[args.rnn_type], padding=args.rnn_padding,)
        doc_hidden_size = 2 * args.hidden_size
        question_hidden_size = 2 * args.hidden_size
        if args.concat_rnn_layers:
            doc_hidden_size *= args.doc_layers
            question_hidden_size *= args.question_layers
        if args.question_merge not in ['avg', 'self_attn']:raise NotImplementedError
        if args.question_merge == 'self_attn':self.self_attn = LinearSeqAttn(question_hidden_size)
        self.start_attn = BilinearSeqAttn(doc_hidden_size,question_hidden_size,normalize=normalize,)
        self.end_attn = BilinearSeqAttn(doc_hidden_size,question_hidden_size,normalize=normalize,)
    def forward(self, x1, x1_f, x1_mask, x2, x2_mask):
        x1_emb, x2_emb = self.embedding(x1), self.embedding(x2)
        if self.args.dropout_emb > 0: x1_emb, x2_emb = dropout(x1_emb, p=self.args.dropout_emb, training=self.training), dropout(x2_emb, p=self.args.dropout_emb, training=self.training)
        drnn_input = [x1_emb]
        if self.args.use_qemb: drnn_input.append(self.qemb_match(x1_emb, x2_emb, x2_mask))
        if self.args.num_features > 0: drnn_input.append(x1_f)
        doc_hiddens, question_hiddens = self.doc_rnn(cat(drnn_input, 2), x1_mask), self.question_rnn(x2_emb, x2_mask)
        if self.args.question_merge == 'avg':
            alpha = ones(question_hiddens.size(0), question_hiddens.size(1)) * x2_mask.eq(0).float()
            q_merge_weights = alpha / alpha.sum(1).expand(alpha.size())
        elif self.args.question_merge == 'self_attn': q_merge_weights = self.self_attn(question_hiddens, x2_mask)
        question_hidden = q_merge_weights.unsqueeze(1).bmm(question_hiddens).squeeze(1)
        return self.start_attn(doc_hiddens, question_hidden, x1_mask), self.end_attn(doc_hiddens, question_hidden, x1_mask)
class DocReader(object):
    def __init__(self, args, word_dict, feature_dict, state_dict=None, normalize=False):
        self.args, self.word_dict, self.args.vocab_size, self.feature_dict, self.args.num_features, self.updates, self.parallel, self.network = args, word_dict, len(word_dict), feature_dict, len(feature_dict), 0, False, RnnDocReader(args, normalize)
        if state_dict:
            if 'fixed_embedding' in state_dict:
                fixed_embedding = state_dict.pop('fixed_embedding')
                self.network.load_state_dict(state_dict)
                self.network.register_buffer('fixed_embedding', fixed_embedding)
            else: self.network.load_state_dict(state_dict)
    def expand_dictionary(self, words):
        to_add = {self.word_dict.normalize(w) for w in words if w not in self.word_dict}
        if to_add:
            for w in to_add: self.word_dict.add(w)
            self.args.vocab_size = len(self.word_dict)
            old_embedding = self.network.embedding.weight.data
            self.network.embedding = Embedding(self.args.vocab_size, self.args.embedding_dim, padding_idx=0)
            new_embedding = self.network.embedding.weight.data
            new_embedding[:old_embedding.size(0)] = old_embedding
        return to_add
    def load_embeddings(self, words, embedding_file):
        words, embedding, vec_counts = {w for w in words if w in self.word_dict}, self.network.embedding.weight.data, {}
        with open(embedding_file) as f:
            for line in f:
                parsed = line.rstrip().split(' ')
                assert(len(parsed) == embedding.size(1) + 1)
                w = self.word_dict.normalize(parsed[0])
                if w in words:
                    vec = Tensor([float(i) for i in parsed[1:]])
                    if w not in vec_counts:
                        vec_counts[w] = 1
                        embedding[self.word_dict[w]].copy_(vec)
                    else:
                        vec_counts[w] += 1
                        embedding[self.word_dict[w]].add_(vec)
        for w, c in vec_counts.items(): embedding[self.word_dict[w]].div_(c)
    def tune_embeddings(self, words):
        words = {w for w in words if w in self.word_dict}
        if not words or len(words) == len(self.word_dict): return
        embedding = self.network.embedding.weight.data
        for idx, swap_word in enumerate(words, self.word_dict.START):
            curr_word, curr_emb, old_idx = self.word_dict[idx], embedding[idx].clone(), self.word_dict[swap_word]
            embedding[idx].copy_(embedding[old_idx])
            embedding[old_idx].copy_(curr_emb)
            self.word_dict[swap_word], self.word_dict[idx], self.word_dict[curr_word], self.word_dict[old_idx] = idx, swap_word, old_idx, curr_word
        self.network.register_buffer('fixed_embedding', embedding[idx + 1:].clone())
    def init_optimizer(self, state_dict=None):
        if self.args.fix_embeddings:
            for p in self.network.embedding.parameters(): p.requires_grad = False
        parameters = [p for p in self.network.parameters() if p.requires_grad]
        self.optimizer = SGD(parameters, self.args.learning_rate, momentum=self.args.momentum, weight_decay=self.args.weight_decay) if self.args.optimizer == 'sgd' else Adamax(parameters, weight_decay=self.args.weight_decay)
    def update(self, ex):
        if not self.optimizer: raise RuntimeError
        self.network.train()
        inputs = [e if e is None else e for e in ex[:5]]
        target_s, target_e = ex[5:7]
        score_s, score_e = self.network(*inputs)
        loss = nll_loss(score_s, target_s) + nll_loss(score_e, target_e)
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm(self.network.parameters(), self.args.grad_clipping)
        self.optimizer.step()
        self.updates += 1
        self.reset_parameters()
        return loss.data[0], ex[0].size(0)
    def reset_parameters(self):
        if self.args.tune_partial > 0:
            embedding = self.network.module.embedding.weight.data if self.parallel else self.network.embedding.weight.data
            fixed_embedding = self.network.module.fixed_embedding if self.parallel else self.network.fixed_embedding
            offset = embedding.size(0) - fixed_embedding.size(0)
            if offset >= 0: embedding[offset:] = fixed_embedding
    def predict(self, ex, candidates=None, top_n=1, async_pool=None):
        self.network.eval()
        inputs = [e if e is None else Variable(e, volatile=True) for e in ex[:5]]
        score_s, score_e = self.network(*inputs)
        score_s, score_e = score_s.data.cpu(), score_e.data.cpu()
        args ,decode = (score_s, score_e, candidates, top_n, self.args.max_len) if candidates else (score_s, score_e, top_n, self.args.max_len), self.decode_candidates if candidates else self.decode
        return async_pool.apply_async(decode, args) if async_pool else decode(*args)
    @staticmethod
    def decode(score_s, score_e, top_n=1, max_len=None):
        pred_s, pred_e, pred_score = [], [], []
        max_len = max_len or score_s.size(1)
        for i in range(score_s.size(0)):
            scores = ger(score_s[i], score_e[i])
            scores.triu_().tril_(max_len - 1)
            scores = scores.numpy()
            scores_flat = scores.flatten()
            if top_n == 1: idx_sort = [argmax(scores_flat)]
            elif len(scores_flat) < top_n: idx_sort = argsort(-scores_flat)
            else:
                idx = argpartition(-scores_flat, top_n)[0:top_n]
                idx_sort = idx[argsort(-scores_flat[idx])]
            s_idx, e_idx = unravel_index(idx_sort, scores.shape)
            pred_s.append(s_idx)
            pred_e.append(e_idx)
            pred_score.append(scores_flat[idx_sort])
        return pred_s, pred_e, pred_score
    @staticmethod
    def decode_candidates(score_s, score_e, candidates, top_n=1, max_len=None):
        pred_s, pred_e, pred_score = [], [], []
        for i in range(score_s.size(0)):
            tokens, cands = candidates[i]['input'], candidates[i]['cands']
            if not cands: cands = PROCESS_CANDS
            if not cands: raise RuntimeError
            max_len = max_len or len(tokens)
            scores, s_idx, e_idx = [], [], []
            for s, e in tokens.ngrams(n=max_len, as_strings=False):
                span = tokens.slice(s, e).untokenize()
                if span in cands or span.lower() in cands:
                    scores.append(score_s[i][s] * score_e[i][e - 1])
                    s_idx.append(s)
                    e_idx.append(e - 1)
            if len(scores) == 0:
                pred_s.append([])
                pred_e.append([])
                pred_score.append([])
            else:
                scores, s_idx, e_idx = array(scores), array(s_idx), array(e_idx)
                idx_sort = argsort(-scores)[0:top_n]
                pred_s.append(s_idx[idx_sort])
                pred_e.append(e_idx[idx_sort])
                pred_score.append(scores[idx_sort])
        return pred_s, pred_e, pred_score
    def save(self, filename):
        network = self.network.module if self.parallel else self.network
        state_dict = copy(network.state_dict())
        if 'fixed_embedding' in state_dict: state_dict.pop('fixed_embedding')
        try: save({'state_dict': state_dict, 'word_dict': self.word_dict, 'feature_dict': self.feature_dict, 'args': self.args,}, filename)
        except: pass
    def checkpoint(self, filename, epoch):
        try: save({'state_dict': self.network.module.state_dict() if self.parallel else self.network.state_dict(), 'word_dict': self.word_dict, 'feature_dict': self.feature_dict, 'args': self.args, 'epoch': epoch, 'optimizer': self.optimizer.state_dict(),}, filename)
        except: pass
    @staticmethod
    def load(filename, new_args=None, normalize=True):
        saved_params = tlod(filename, map_location = device('cpu'))
        return DocReader(override_model_args(args, new_args) if new_args else saved_params['args'], saved_params['word_dict'], saved_params['feature_dict'], saved_params['state_dict'], normalize)
    @staticmethod
    def load_checkpoint(filename, normalize=True):
        saved_params = tlod(filename, map_location = device('cpu'))
        model = DocReader(saved_params['args'], saved_params['word_dict'], saved_params['feature_dict'], saved_params['state_dict'], normalize)
        model.init_optimizer(saved_params['optimizer'])
        return model, saved_params['epoch']
    def cpu(self): self.network = self.network.cpu()
    def parallelize(self): self.parallel, self.network = True, DataParallel(self.network)
class Tokens(object):
    def __init__(self, data, annotators, opts=None): self.data, self.annotators, self.opts = data, annotators, opts or {}
    def __len__(self): return len(self.data)
    def __bool__(self): return bool(self.data)
    def slice(self, i=None, j=None):
        new_tokens = copy(self)
        new_tokens.data = self.data[i: j]
        return new_tokens
    def untokenize(self): return ''.join([t[1] for t in self.data]).strip()
    def words(self, uncased=False): return [t[0].lower() for t in self.data] if uncased else [t[0] for t in self.data]
    def offsets(self): return [t[2] for t in self.data]
    def pos(self): return None if 'pos' not in self.annotators else [t[3] for t in self.data]
    def lemmas(self): return None if 'lemma' not in self.annotators else [t[4] for t in self.data]
    def entities(self): return None if 'ner' not in self.annotators else [t[5] for t in self.data]
    def ngrams(self, n=1, uncased=False, filter_fn=None, as_strings=True):
        def _skip(gram): return False if not filter_fn else filter_fn(gram)
        words = self.words(uncased)
        ngrams = [(s, e + 1) for s in range(len(words)) for e in range(s, min(s + n, len(words))) if not _skip(words[s:e + 1])]
        return ['{}'.format(' '.join(words[s:e])) for (s, e) in ngrams] if as_strings else ngrams
    def entity_groups(self):
        entities = self.entities()
        if not entities: return None
        non_ent, groups, idx = self.opts.get('non_ent', 'O'), [], 0
        while idx < len(entities):
            ner_tag = entities[idx]
            if ner_tag != non_ent:
                start = idx
                while (idx < len(entities) and entities[idx] == ner_tag): idx += 1
                groups.append((self.slice(start, idx).untokenize(), ner_tag))
            else: idx += 1
        return groups
class DocDB(object):
    def __init__(self, db_path=None):
        self.path = db_path or '.\\data\\wikipedia\\docs.db'
        self.connection = connect(self.path, check_same_thread=False)
    def __enter__(self): return self
    def __exit__(self, *args): self.close()
    def path(self): return self.path
    def close(self): self.connection.close()
    def get_doc_ids(self):
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM documents")
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results
    def get_doc_text(self, doc_id):
        cursor = self.connection.cursor()
        cursor.execute("SELECT text FROM documents WHERE id = ?", (normalize('NFD',doc_id),))
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else result[0]
class Tokenizer(object):
    def tokenize(self, text): raise NotImplementedError
    def shutdown(self): pass
    def __del__(self): self.shutdown()
class CoreNLPTokenizer(Tokenizer):
    def __init__(self, **kwargs):
        self.classpath = (kwargs.get('classpath') or '.\\data\\corenlp\\*')
        self.annotators = deepcopy(kwargs.get('annotators', set()))
        self.mem = kwargs.get('mem', '2g')
        self._launch()
    def _launch(self):
        annotators = 'tokenize,ssplit'
        if 'ner' in self.annotators: annotators += ',pos,lemma,ner'
        elif 'lemma' in self.annotators: annotators += ',pos,lemma'
        elif 'pos' in self.annotators: annotators += ',pos'
        cmd = ['java', '-mx' + self.mem, '-cp', '"%s"' % self.classpath, 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', annotators, '-tokenize.options', 'untokenizable=noneDelete,invertible=true', '-outputFormat', 'json', '-prettyPrint', 'false']
        self.corenlp = spawn('/bin/bash', maxread=100000, timeout=60)
        self.corenlp.setecho(False)
        self.corenlp.sendline('stty -icanon')
        self.corenlp.sendline(' '.join(cmd))
        self.corenlp.delaybeforesend = 0
        self.corenlp.delayafterread = 0
        self.corenlp.expect_exact('NLP>', searchwindowsize=100)
    @staticmethod
    def _convert(token): return token.replace('-LRB-', '(').replace('-RRB-', ')').replace('-LSB-', '[').replace('-RSB-', '[').replace('-LCB-', '{').replace('-RCB-', '}') if len(token) == 5 else token
    def tokenize(self, text):
        if 'NLP>' in text: raise RuntimeError
        if text.lower().strip() == 'q':
            token = text.strip()
            index = text.index(token)
            return Tokens([(token, text[index:], (index, index + 1), 'NN', 'q', 'O')], self.annotators)
        self.corenlp.sendline(text.replace('\n', ' ').encode('utf-8'))
        self.corenlp.expect_exact('NLP>', searchwindowsize=100)
        output = self.corenlp.before
        output = loads(output[output.find(b'{"sentences":'):].decode('utf-8'))
        data = []
        tokens = [t for s in output['sentences'] for t in s['tokens']]
        for i in range(len(tokens)): data.append((self._convert(tokens[i]['word']),  text[tokens[i]['characterOffsetBegin']: tokens[i + 1]['characterOffsetBegin'] if i + 1 < len(tokens) else tokens[i]['characterOffsetEnd']],(tokens[i]['characterOffsetBegin'], tokens[i]['characterOffsetEnd']), tokens[i].get('pos', None), tokens[i].get('lemma', None), tokens[i].get('ner', None)))
        return Tokens(data, self.annotators)
class RegexpTokenizer(Tokenizer):
    def __init__(self, **kwargs):self._regexp, self.annotators, self.substitutions = compile(r'(?P<digit>\p{Nd}+([:\.\,]\p{Nd}+)*)|(?P<title>(dr|esq|hon|jr|mr|mrs|ms|prof|rev|sr|st|rt|messrs|mmes|msgr)\.(?=\p{Z}))|(?P<abbr>([\p{L}]\.){2,}(?=\p{Z}|$))|(?P<neg>%s)|(?P<hyph>[\p{L}\p{N}\p{M}]++([-\u058A\u2010\u2011][\p{L}\p{N}\p{M}]++)+)|(?P<contr1>can(?=not\b))|(?P<alphanum>[\p{L}\p{N}\p{M}]++)|(?P<contr2>%s)|(?P<sdquote>(?<=[\p{Z}\(\[{<]|^)(``|["\u0093\u201C\u00AB])(?!\p{Z}))|(?P<edquote>(?<!\p{Z})(\'\'|["\u0094\u201D\u00BB]))|(?P<ssquote>(?<=[\p{Z}\(\[{<]|^)[\'\u0091\u2018\u201B\u2039](?!\p{Z}))|(?P<esquote>(?<!\p{Z})[\'\u0092\u2019\u203A])|(?P<dash>--|[\u0096\u0097\u2013\u2014\u2015])|(?<ellipses>\.\.\.|\u2026)|(?P<punct>\p{P})|(?P<nonws>[^\p{Z}\p{C}])' % (r"((?!n't)[\p{L}\p{N}\p{M}])++(?=n't)|n't", r"'([tsdm]|re|ll|ve)\b"), flags=42), set(), kwargs.get('substitutions', True)
    def tokenize(self, text):
        data = []
        matches = [m for m in self._regexp.finditer(text)]
        for i in range(len(matches)):
            token = matches[i].group()
            if self.substitutions:
                groups = matches[i].groupdict()
                if groups['sdquote']: token = "``"
                elif groups['edquote']: token = "''"
                elif groups['ssquote']: token = "`"
                elif groups['esquote']: token = "'"
                elif groups['dash']: token = '--'
                elif groups['ellipses']: token = '...'
            span = matches[i].span() 
            data.append((token, text[span[0]: matches[i + 1].span()[0] if i + 1 < len(matches) else span[1]], span))
        return Tokens(data, self.annotators)
class SpacyTokenizer(Tokenizer):
    def __init__(self, **kwargs):
        model = kwargs.get('model', 'en')
        self.annotators = deepcopy(kwargs.get('annotators', set()))
        nlp_kwargs = {'parser': False}
        if not any([p in self.annotators for p in ['lemma', 'pos', 'ner']]): nlp_kwargs['tagger'] = False
        if 'ner' not in self.annotators: nlp_kwargs['entity'] = False
        self.nlp = load(model, **nlp_kwargs)
    def tokenize(self, text):
        clean_text = text.replace('\n', ' ')
        tokens = self.nlp.tokenizer(clean_text)
        if any([p in self.annotators for p in ['lemma', 'pos', 'ner']]): self.nlp.tagger(tokens)
        if 'ner' in self.annotators: self.nlp.entity(tokens)
        data = []
        for i in range(len(tokens)): data.append((tokens[i].text, text[tokens[i].idx: tokens[i + 1].idx if i + 1 < len(tokens) else tokens[i].idx + len(tokens[i].text)], (tokens[i].idx, tokens[i].idx + len(tokens[i].text)), tokens[i].tag_, tokens[i].lemma_, tokens[i].ent_type_,))
        return Tokens(data, self.annotators, opts={'non_ent': ''})
class SimpleTokenizer(Tokenizer):
    def __init__(self, **kwargs):self._regexp, self.annotators = compile(r'[\p{L}\p{N}\p{M}]|[^\p{Z}\p{C}]', flags=42), set()
    def tokenize(self, text):
        data, matches = [], [m for m in self._regexp.finditer(text)]
        for i in range(len(matches)):
            span = matches[i].span()
            data.append((matches[i].group(), text[span[0]: matches[i + 1].span()[0] if i + 1 < len(matches) else span[1]], span,))
        return Tokens(data, self.annotators)
class SortedBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle=True): self.lengths, self.batch_size, self.shuffle = lengths, batch_size, shuffle
    def __iter__(self):
        lengths = array([(-l[0], -l[1], random()) for l in self.lengths], dtype=[('l1', int_), ('l2', int_), ('rand', float_)])
        indices = argsort(lengths, order=('l1', 'l2', 'rand'))
        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        if self.shuffle: shuffle(batches)
        return iter([i for batch in batches for i in batch])
    def __len__(self): return len(self.lengths)
class ReaderDataset(Dataset):
    def __init__(self, examples, model, single_answer=False):self.model, self.examples, self.single_answer = model, examples, single_answer
    def __len__(self): return len(self.examples)
    def __getitem__(self, index): return vectorize(self.examples[index], self.model, self.single_answer)
    def lengths(self): return [(len(ex['document']), len(ex['question'])) for ex in self.examples]
class TfidfDocRanker(object):
    def __init__(self, tfidf_path=None, strict=True):
        tfidf_path = tfidf_path or '.\\data\\wikipedia\\docs-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz'
        matrix, metadata = load_sparse_csr(tfidf_path)
        self.doc_mat = matrix
        self.ngrams = metadata['ngram']
        self.hash_size = metadata['hash_size']
        self.tokenizer = get_class(metadata['tokenizer'])()
        self.doc_freqs = metadata['doc_freqs'].squeeze()
        self.doc_dict = metadata['doc_dict']
        self.num_docs = len(self.doc_dict[0])
        self.strict = strict
    def get_doc_id(self, doc_index): return self.doc_dict[1][doc_index]
    def closest_docs(self, query, k=1):
        res = self.text2spvec(query) * self.doc_mat
        if len(res.data) <= k: o_sort = argsort(-res.data)
        else:
            o = argpartition(-res.data, k)[0:k]
            o_sort = o[argsort(-res.data[o])]
        return [self.get_doc_id(i) for i in res.indices[o_sort]], res.data[o_sort]
    def parse(self, query): return self.tokenizer.tokenize(query).ngrams(n=self.ngrams, uncased=True, filter_fn=filter_ngram)
    def text2spvec(self, query):
        words = self.parse(normalize('NFD', query))
        wids = [murmurhash3_32(w, positive=True) % self.hash_size for w in words]
        if not wids and self.strict: raise RuntimeError
        elif not wids: return csr_matrix((1, self.hash_size))
        wids_unique, wids_counts = unique(wids, return_counts=True)
        Ns = self.doc_freqs[wids_unique]
        idfs = log((self.num_docs - Ns + 0.5) / (Ns + 0.5))
        idfs[idfs < 0] = 0
        return csr_matrix((multiply(log1p(wids_counts), idfs), wids_unique, array([0, len(wids_unique)])), shape=(1, self.hash_size))
class QA(object):
    def __init__(self, fixed_candidates=None, ranker_config=None, db_config=None):
        self.fixed_candidates = fixed_candidates is not None
        self.candidates, ranker_config = fixed_candidates or {}, ranker_config or {}
        ranker_class = ranker_config.get('class', TfidfDocRanker)
        ranker_opts = ranker_config.get('options', {})
        self.ranker = ranker_class(**ranker_opts)
        self.reader = DocReader.load('.\\data\\reader\\multitask.mdl')
        db_config = db_config or {}
        self.processes = Pool(None, initializer=init, initargs=(CoreNLPTokenizery, {'annotators': gafa(self.reader.args)}, db_config.get('class', DocDB), db_config.get('options', {}), fixed_candidates))
    def _split_doc(self, doc):
        curr, curr_len = [], 0
        for split in split(r'\n+', doc):
            split = split.strip()
            if curr and (curr_len or split):
                yield ' '.join(curr)
                curr, curr_len = [], 0
            if split:
                curr.append(split)
                curr_len += len(split)
        if curr: yield ' '.join(curr)
    def _get_loader(self, data, num_loaders):
        dataset = ReaderDataset(data, self.reader)
        return DataLoader(dataset, batch_size=128, sampler=SortedBatchSampler(dataset.lengths(), 128, shuffle=False), num_workers=num_loaders, collate_fn=batchify, pin_memory=False,)
    def process(self, query, candidates=None):
        ranked = [self.ranker.closest_docs(query, k=5)]
        all_docids, all_doc_scores = zip(*ranked)
        flat_docids = list({d for docids in all_docids for d in docids})
        did2didx = {did: didx for didx, did in enumerate(flat_docids)}
        doc_texts = self.processes.map(fetch_text, flat_docids)
        flat_splits, didx2sidx = [], []
        for text in doc_texts:
            splits = self._split_doc(text)
            didx2sidx.append([len(flat_splits), -1])
            for split in splits:flat_splits.append(split)
            didx2sidx[-1][1] = len(flat_splits)
        q_tokens = self.processes.map_async(tokenize_text, [query]).get()
        s_tokens = self.processes.map_async(tokenize_text, flat_splits).get()
        examples = []
        for rel_didx, did in enumerate(all_docids[0]):
            start, end = didx2sidx[did2didx[did]]
            for sidx in range(start, end):
                if q_tokens[0].words() and s_tokens[sidx].words(): examples.append({'id': (0, rel_didx, sidx), 'question': q_tokens[0].words(), 'qlemma': q_tokens[0].lemmas(), 'document': s_tokens[sidx].words(), 'lemma': s_tokens[sidx].lemmas(), 'pos': s_tokens[sidx].pos(), 'ner': s_tokens[sidx].entities()})
        result_handles = []
        num_loaders = min(5, math.floor(len(examples) / 1e3))
        for batch in self._get_loader(examples, num_loaders):
            if candidates or self.fixed_candidates:
                batch_cands = []
                for ex_id in batch[-1]: batch_cands.append({'input': s_tokens[ex_id[2]], 'cands': candidates if candidates else None })
                handle = self.reader.predict(batch, batch_cands, async_pool=self.processes)
            else: handle = self.reader.predict(batch, async_pool=self.processes)
            result_handles.append((handle, batch[-1], batch[0].size(0)))
        queues = [[]]
        for result, ex_ids, batch_size in result_handles:
            s, e, score = result.get()
            for i in range(batch_size):
                if score[i]:
                    item = (score[i][0], ex_ids[i], s[i][0], e[i][0])
                    queue = queues[ex_ids[i][0]]
                    if not queue:heapq.heappush(queue, item)
                    else:heapq.heappushpop(queue, item)
        predictions = []
        while len(queues[0]) > 0:
            score, (qidx, rel_didx, sidx), s, e = heapq.heappop(queues[0])
            prediction = {'doc_id': all_docids[qidx][rel_didx], 'span': s_tokens[sidx].slice(s, e + 1).untokenize(), 'doc_score': float(all_doc_scores[qidx][rel_didx]), 'span_score': float(score),}
            prediction['context'] = {'text': s_tokens[sidx].untokenize(), 'start': s_tokens[sidx].offsets()[s][0], 'end': s_tokens[sidx].offsets()[e][1],}
            predictions.append(prediction)
        return predictions[-1::-1]
    def answer(question):
        predictions, table = self.process(question, self.candidates), PrettyTable(['Rank', 'Answer', 'Doc', 'Answer Score', 'Doc Score'])
        for i, p in enumerate(predictions, 1):table.add_row([i, p['span'], p['doc_id'],'%.5g' % p['span_score'],'%.5g' % p['doc_score']])
        final_text = f'Top Predictions:\n{table}\n\nContexts:'
        for p in predictions:
            text, start, end = p['context']['text'], p['context']['start'], p['context']['end']
            output = (text[:start] + colored(text[start: end], 'green', attrs=['bold']) + text[end:])
            final_text += '\n[ Doc = %s ]\n%s\n' % (p['doc_id'], output)
        return final_text
def override_model_args(old_args, new_args):
    old_args, new_args = vars(old_args), vars(new_args)
    for k in old_args.keys():
        if k in new_args and old_args[k] != new_args[k] and k in {'fix_embeddings', 'optimizer', 'learning_rate', 'momentum', 'weight_decay','rnn_padding', 'dropout_rnn', 'dropout_rnn_output', 'dropout_emb','max_len', 'grad_clipping', 'tune_partial'}: old_args[k] = new_args[k]
    return Namespace(**old_args)
def vectorize(ex, model, single_answer=False):
    args = model.args
    word_dict = model.word_dict
    feature_dict = model.feature_dict
    document = LongTensor([word_dict[w] for w in ex['document']])
    question = LongTensor([word_dict[w] for w in ex['question']])
    features = zeros(len(ex['document']), len(feature_dict)) if feature_dict else None
    if args.use_in_question:
        for i in range(len(ex['document'])):
            if ex['document'][i] in {w for w in ex['question']}: features[i][feature_dict['in_question']] = 1.0
            if ex['document'][i].lower() in {w.lower() for w in ex['question']}: features[i][feature_dict['in_question_uncased']] = 1.0
            if args.use_lemma and ex['lemma'][i] in {w for w in ex['qlemma']}: features[i][feature_dict['in_question_lemma']] = 1.0
    if args.use_pos:
        for i, w in enumerate(ex['pos']):
            f = 'pos=' + w
            if f in feature_dict: features[i][feature_dict[f]] = 1.0
    if args.use_ner:
        for i, w in enumerate(ex['ner']):
            f = 'ner=' + w
            if f in feature_dict: features[i][feature_dict[f]] = 1.0
    if args.use_tf:
        for i, w in enumerate(ex['document']): features[i][feature_dict['tf']] = Counter([w.lower() for w in ex['document']])[w.lower()] * 1.0 / len(ex['document'])
    if 'answers' not in ex: return document, features, question, ex['id']
    if single_answer:
        assert(len(ex['answers']) > 0)
        start, end = LongTensor(1).fill_(ex['answers'][0][0]), LongTensor(1).fill_(ex['answers'][0][1])
    else: start, end = [a[0] for a in ex['answers']], [a[1] for a in ex['answers']]
    return document, features, question, start, end, ex['id']
def batchify(batch):
    NUM_INPUTS = 3
    NUM_TARGETS = 2
    NUM_EXTRA = 1
    ids = [ex[-1] for ex in batch]
    docs = [ex[0] for ex in batch]
    features = [ex[1] for ex in batch]
    questions = [ex[2] for ex in batch]
    max_length = max([d.size(0) for d in docs])
    x1 = LongTensor(len(docs), max_length).zero_()
    x1_mask = ByteTensor(len(docs), max_length).fill_(1)
    if features[0] is None: x1_f = None
    else: x1_f = zeros(len(docs), max_length, features[0].size(1))
    for i, d in enumerate(docs):
        x1[i, :d.size(0)].copy_(d)
        x1_mask[i, :d.size(0)].fill_(0)
        if x1_f is not None:  x1_f[i, :d.size(0)].copy_(features[i])
    max_length = max([q.size(0) for q in questions])
    x2 = LongTensor(len(questions), max_length).zero_()
    x2_mask = ByteTensor(len(questions), max_length).fill_(1)
    for i, q in enumerate(questions):
        x2[i, :q.size(0)].copy_(q)
        x2_mask[i, :q.size(0)].fill_(0)
    if len(batch[0]) == NUM_INPUTS + NUM_EXTRA: return x1, x1_f, x1_mask, x2, x2_mask, ids
    elif len(batch[0]) == NUM_INPUTS + NUM_EXTRA + NUM_TARGETS:
        if is_tensor(batch[0][3]):
            y_s = cat([ex[3] for ex in batch])
            y_e = cat([ex[4] for ex in batch])
        else:
            y_s = [ex[3] for ex in batch]
            y_e = [ex[4] for ex in batch]
    return x1, x1_f, x1_mask, x2, x2_mask, y_s, y_e, ids
def load_sparse_csr(filename):
    loader = nlod(filename, allow_pickle=True)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape']), loader['metadata'].item(0) if 'metadata' in loader else None
def get_class(name):
    if name == 'spacy': return SpacyTokenizer
    if name == 'corenlp': return CoreNLPTokenizer
    if name == 'regexp': return RegexpTokenizer
    if name == 'simple': return SimpleTokenizer
def filter_ngram(gram, mode='any'):
    if mode == "all":return all([match(r'^\p{P}+$', normalize('NFD', w)) or normalize('NFD', w).lower() in Wordlist for w in gram])
    if mode == 'any':return any([match(r'^\p{P}+$', normalize('NFD', w)) or normalize('NFD', w).lower() in Wordlist for w in gram])
    else:return match(r'^\p{P}+$', normalize('NFD', gram[0])) or normalize('NFD', gram[0]).lower() in Wordlist or match(r'^\p{P}+$', normalize('NFD', gram[-1])) or normalize('NFD', gram[-1]).lower() in Wordlist
def init(tokenizer_class, tokenizer_opts, db_class, db_opts, candidates=None):
    global PROCESS_TOK, PROCESS_DB, PROCESS_CANDS
    PROCESS_TOK = tokenizer_class(**tokenizer_opts)
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)
    PROCESS_CANDS = candidates
def fetch_text(doc_id):
    global PROCESS_DB
    return PROCESS_DB.get_doc_text(doc_id)
def tokenize_text(text):
    global PROCESS_TOK
    return PROCESS_TOK.tokenize(text)
def gafa(args):
    annotators = set()
    if args.use_pos:annotators.add('pos')
    if args.use_lemma:annotators.add('lemma')
    if args.use_ner:annotators.add('ner')
    return annotators
def my_hook(t):
    last_b = [0]
    def update_to(b=1, bsize=1, tsize=None):
        if tsize is not None: t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b
    return update_to
def _torch(): pass # generate the reader folder, I need to write this code...
def setup():
    if not isdir('.\\data') and 'n' in input('I\'m going to install Wikipedia, this may take a long time (this is 7 Gb) would you like to do in in another time? (y/n)\n').lower():
        with tqdm('http://gfs270n122.userstorage.mega.co.nz/dl/RVLMhIxMApeyQQIxkFDWl-0aHd_iqz6yDLq4wI2brHvPeHgy_D9mdO7470RrDQwek4XjBUzF0Nc8SlxvUWPoM8hjTJAFUQig6jgP8dP6udLY-3_4bwfmcMs7HcC_Pg', desc = 'Installing Wikipedia') as t:
            reporthook = my_hook(t)
            urlretrieve('http://gfs270n122.userstorage.mega.co.nz/dl/RVLMhIxMApeyQQIxkFDWl-0aHd_iqz6yDLq4wI2brHvPeHgy_D9mdO7470RrDQwek4XjBUzF0Nc8SlxvUWPoM8hjTJAFUQig6jgP8dP6udLY-3_4bwfmcMs7HcC_Pg', '.\\data.tar.gz', reporthook=reporthook)
        with topen('.\\data.tar.gz', 'r:gz') as tar:
            for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers()), desc = 'Extracting wikipedia'): tar.extract(member=member, path = '.')
        remove('.\\data.tar.gz')
        _torch()
        print('DONE\n')
    elif not isdir('.\\data'): exit(print('You chose not to install wikipedia, you can re-run it any time and install wikipedia'))
    if not isfile('data.in'): open('data.in', 'a').write(input('Give me a candidate file, if you don\'t have any press enter\n'))
def question(question):
    try: print(QA(set(normalize('NFD', line.strip()).lower() for line in open(open('data.in').read().replace('\n', ''))) if open('data.in', 'r').read().replace('\n', '') else None, {'options': {'tfidf_path': '.\\data\\wikipedia\\docs-tfidf-ngram=2-hash=16777216-tokenizer=simple.npz'}}, {'options': {'db_path': '.\\data\\wikipedia\\docs.db'}}).answer(question))
    except: print('The process failed, have you ran the setup function yet?')
def interactive():
    while True:
        query = input('What do you want to know? (just press Enter without input to exit the interactive session)\n')
        if query: question(query)
        else: break

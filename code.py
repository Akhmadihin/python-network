import torch # pip install torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict, Counter # pip install collections-extended
import numpy as np # pip install numpy
import random # pip install random2
import re # pip install regex
import os # pip install os-sys
from datetime import datetime # pip install datetime

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout): # create a network
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, 
                           num_layers=num_layers, 
                           dropout=dropout if num_layers > 1 else 0,
                           batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
    def forward(self, x, hidden=None):
        # сhecking indexes before embedding
        if torch.any(x >= self.embedding.num_embeddings):
            print(f"Обнаружен недопустимый индекс: {torch.max(x)} (размер словаря: {self.embedding.num_embeddings})")
            x = torch.clamp(x, 0, self.embedding.num_embeddings-1)
            
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        if hidden is None:
            batch_size = x.size(0)
            hidden = self.init_hidden(batch_size)
            
        output, hidden = self.lstm(embedded, hidden)
        output = self.dropout(output)
        logits = self.fc(output)
        return logits, hidden
        
    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))

class SmartLanguageModel:
    def __init__(self):
        # dictionary initialization
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}
        self.word_counts = Counter()
        
        # model parameters
        self.embedding_dim = 512
        self.hidden_size = 1024
        self.num_layers = 5
        self.dropout = 0.3
        self.seq_length = 30
        self.batch_size = 32
        
        # first, we load the data and update the dictionary
        self._load_data_and_build_vocab()
        
        # then, initialize the model with the correct dictionary size
        self.model = LanguageModel(
            self.vocab_size,
            self.embedding_dim,
            self.hidden_size,
            self.num_layers,
            self.dropout
        )
        
        self.length_predictor = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.optimizer = optim.Adam(
            list(self.model.parameters()) + list(self.length_predictor.parameters()),
            lr=0.001
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.length_criterion = nn.MSELoss()
        
        self.conversation_history = []
        
        if os.path.exists('language_model.pth'):
            self.load_model('language_model.pth')
            self.interactive_chat()

    def _load_data_and_build_vocab(self):
        """Загружает данные и строит словарь"""
        if os.path.exists('train.txt'):
            with open('train.txt', 'r', encoding='utf-8') as f:
                text = f.read()
            
            tokens = self.preprocess_text(text)
            
            # first, we count the frequency of words
            word_counts = Counter(tokens)
            
            # adding words to the dictionary
            for word, count in word_counts.items():
                if count >= 1 and word not in self.word_to_idx:
                    idx = len(self.word_to_idx)
                    self.word_to_idx[word] = idx
                    self.idx_to_word[idx] = word
            
            self.vocab_size = len(self.word_to_idx)
            print(f"Размер словаря после обработки данных: {self.vocab_size}")
        else:
            self.vocab_size = len(self.word_to_idx)
            print("Файл train.txt не найден, использую базовый словарь")

    def _build_model(self):
        """create a language model"""
        print(f"Creating a model with vocab_size={self.vocab_size}")
        return LanguageModel(
            self.vocab_size,
            self.embedding_dim,
            self.hidden_size,
            self.num_layers,
            self.dropout
        )
    def _build_length_predictor(self):
        """Creates a model for predicting the length of a response"""
        return nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def preprocess_text(self, text):
        text = re.sub(r'[^\w\s.,!?-]', ' ', text.lower())
        return re.findall(r"\w+|[.,!?]", text)

    def text_to_indices(self, text, max_length=None):
        tokens = self.preprocess_text(text)
        indices = [self.word_to_idx.get(token, self.word_to_idx['<UNK>']) for token in tokens]
        return indices[:max_length] if max_length else indices

    def train_model(self, sequences, target_lengths, epochs=50):
        print(f"The beginning of training. Dictionary Size: {self.vocab_size}")
        
        # Checking indexes
        max_idx = max(max(seq[0] + seq[1]) for seq in sequences)
        if max_idx >= self.vocab_size:
            print(f"Mistake: Maximum index {max_idx} >= dictionary size {self.vocab_size}")
            print("Increase the minimum word frequency or add more data")
            return

        epochs=50
        for epoch in range(epochs):
            total_loss = 0
            length_loss = 0
            
            indices = list(range(len(sequences)))
            random.shuffle(indices)
            
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i+self.batch_size]
                batch_sequences = [sequences[idx] for idx in batch_indices]
                batch_lengths = [target_lengths[idx] for idx in batch_indices]
                
                inputs = [torch.tensor(seq[0][:self.seq_length], dtype=torch.long) for seq in batch_sequences]
                targets = [torch.tensor(seq[1][:self.seq_length], dtype=torch.long) for seq in batch_sequences]
                
                inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
                targets = pad_sequence(targets, batch_first=True, padding_value=0)
                length_targets = torch.tensor(batch_lengths, dtype=torch.float)
                
                self.optimizer.zero_grad()
                
                outputs, hidden = self.model(inputs)
                loss = self.criterion(outputs.view(-1, self.vocab_size), targets.view(-1))
                
                _, (hn, _) = self.model.lstm(self.model.embedding(inputs))
                hn = hn[-1]
                predicted_lengths = self.length_predictor(hn).squeeze()
                l_loss = self.length_criterion(predicted_lengths, length_targets)
                
                total_loss = loss + l_loss
                total_loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                length_loss += l_loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(sequences):.4f}")

    def predict_response_length(self, input_tensor):
        with torch.no_grad():
            _, (hn, _) = self.model.lstm(self.model.embedding(input_tensor))
            hn = hn[-1]
            predicted_length = self.length_predictor(hn).item()
            return max(5, min(100, int(predicted_length * 100 + 0.5)))

    def generate_response(self, input_text, temperature=0.7, top_k=10):
        input_seq = [self.word_to_idx['<SOS>']] + self.text_to_indices(input_text)
        input_tensor = torch.tensor([input_seq[-self.seq_length:]], dtype=torch.long)
        
        predicted_length = self.predict_response_length(input_tensor)
        print(f"Предсказанная длина ответа: {predicted_length} токенов")
        
        generated = []
        hidden = None
        
        with torch.no_grad():
            for _ in range(predicted_length):
                output, hidden = self.model(input_tensor, hidden)
                output = output[0, -1, :] / temperature
                
                if top_k > 0 and top_k < len(output):
                    top_values, _ = torch.topk(output, top_k)
                    output[output < top_values[-1]] = -float('Inf')
                
                probabilities = torch.softmax(output, dim=-1)
                predicted_idx = torch.multinomial(probabilities, 1).item()
                
                if predicted_idx == self.word_to_idx['<EOS>']:
                    break
                
                predicted_word = self.idx_to_word.get(predicted_idx, '<UNK>')
                generated.append(predicted_word)
                input_tensor = torch.tensor([[predicted_idx]], dtype=torch.long)
        
        response = ' '.join(generated)
        response = re.sub(r'\s([.,!?])', r'\1', response)
        return response.capitalize()

    def save_model(self, path):
        torch.save({
            'model_state': self.model.state_dict(),
            'length_predictor_state': self.length_predictor.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'word_counts': dict(self.word_counts)
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.word_to_idx = checkpoint['word_to_idx']
        self.idx_to_word = checkpoint['idx_to_word']
        self.word_counts = Counter(checkpoint['word_counts'])
        self.vocab_size = len(self.word_to_idx)
        
        self.model = self._build_model()
        self.length_predictor = self._build_length_predictor()
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.length_predictor.load_state_dict(checkpoint['length_predictor_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        print(f"The model is loaded, the size of the dictionary: {self.vocab_size}")

    def interactive_chat(self):
        print("Hi! I'm a smart language model. Let's chat! (write 'exit' to end)")
        
        while True:
            try:
                user_input = input("you: ").strip()
                
                if user_input.lower() in ['exit', 'quit']:
                    print("Goodbye! Saving the model...")
                    self.save_model('language_model.pth')
                    break
                
                if not user_input:
                    continue
                
                # processing user input
                user_tokens = self.preprocess_text(user_input)
                self.update_vocabulary(user_tokens)
                self.conversation_history.append(user_input)
                
                # generating a response
                response = self.generate_response(user_input)
                print(f"Бот: {response}")
                self.conversation_history.append(response)
                
                # further education on a new dialogue
                if len(self.conversation_history) >= 2:
                    sequences = []
                    target_lengths = []
                    
                    for i in range(len(self.conversation_history)-1):
                        input_seq = self.text_to_indices(self.conversation_history[i])
                        target_seq = self.text_to_indices(self.conversation_history[i+1])
                        sequences.append((input_seq, target_seq))
                        target_lengths.append(len(target_seq)/100)  # Нормализация длины
                    
                    if sequences:
                        self.train_model(sequences, target_lengths, epochs=1)
                
                # Auto-save every 5 messages
                if len(self.conversation_history) % 5 == 0:
                    self.save_model('language_model.pth')
            
            except KeyboardInterrupt:
                print("\nSaving the model before exiting...")
                self.save_model('language_model.pth')
                break
            except Exception as e:
                print(f"An error has occurred: {e}")
                print("We continue the dialogue...")

    def update_vocabulary(self, tokens):
        """Updates the dictionary with new words"""
        new_words_added = False
        for token in tokens:
            if token not in self.word_to_idx:
                # adding a new word to the dictionary
                idx = len(self.word_to_idx)
                self.word_to_idx[token] = idx
                self.idx_to_word[idx] = token
                self.word_counts[token] = 1
                new_words_added = True
            else:
                self.word_counts[token] += 1
        
        if new_words_added:
            # if you add new words, you need to recreate the model
            self.vocab_size = len(self.word_to_idx)
            print(f"New words have been added. New dictionary size: {self.vocab_size}")
            
            # We keep the current weights
            old_model_state = self.model.state_dict()
            old_predictor_state = self.length_predictor.state_dict()
            
            # recreate the model with a new dictionary size
            self.model = LanguageModel(
                self.vocab_size,
                self.embedding_dim,
                self.hidden_size,
                self.num_layers,
                self.dropout
            )
            
            # restoring weights (new words will receive random weights)
            new_model_state = self.model.state_dict()
            for name, param in old_model_state.items():
                if name in new_model_state:
                    if new_model_state[name].shape == param.shape:
                        new_model_state[name].copy_(param)
            
            self.model.load_state_dict(new_model_state)
            self.length_predictor.load_state_dict(old_predictor_state)
            
            # updating the optimizer
            self.optimizer = optim.Adam(
                list(self.model.parameters()) + list(self.length_predictor.parameters()),
                lr=0.001
            )

if __name__ == "__main__":
    lm = SmartLanguageModel()
    
    if os.path.exists('train.txt'):
        with open('train.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        
        tokens = lm.text_to_indices(text)
        
        sequences = []
        target_lengths = []
        for i in range(0, len(tokens)-lm.seq_length, lm.seq_length//2):
            seq = tokens[i:i+lm.seq_length]
            target = tokens[i+1:i+lm.seq_length+1]
            sequences.append((seq, target))
            target_lengths.append(len(target)/100)
        
        if sequences:
            print(f"Prepared by {len(sequences)} sequences for training")
            lm.train_model(sequences, target_lengths, epochs=3)
    
    lm.interactive_chat()

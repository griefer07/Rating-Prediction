import numpy as np
import tensorflow as tf
import tensorflow.keras.models
from keras import layers
from tensorflow.keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D, Dense, Input, Concatenate, Normalization
import data_cleanup
import pandas as pd
from pathlib import Path


class ProcessData:
    def __init__(self):
        self.exec_path = Path(__file__).resolve().parent

    def tokenize_data(self, max_tokens=3000, seq_length=150, texts=None):
        if texts is None or len(texts) == 0:
            raise ValueError("texts must be a non-empty list of strings")

        # Konvertiere alles zu Strings
        texts = [str(t) for t in texts]

        vec = TextVectorization(
            standardize="lower_and_strip_punctuation",
            split="whitespace",
            max_tokens=max_tokens,
            output_mode="int",
            output_sequence_length=seq_length,
        )
        vec.adapt(texts)
        ids = vec(texts)

        return vec

    def load_data(self, file_path: str, column: int = 1):
        df = pd.read_csv(file_path)
        data = df._get_column_array(column).tolist()
        return data

    def prepare_labels(self, labels):
        """
        Bereitet Labels für 1-5 Klassifikation vor

        Args:
            labels: Liste mit Werten 1-5

        Returns:
            processed_labels: Labels für TensorFlow (0-4)
            num_classes: Anzahl Klassen (5)
        """
        labels = np.array(labels)

        # Prüfe ob Labels wirklich zwischen 1-5 sind
        unique_labels = np.unique(labels)
        print(f"Gefundene Labels: {unique_labels}")

        if not all(label in [1, 2, 3, 4, 5] for label in unique_labels):
            print("Warnung: Nicht alle Labels sind zwischen 1-5!")

        # Konvertiere 1-5 zu 0-4 für TensorFlow
        processed_labels = labels - 1  # 1→0, 2→1, 3→2, 4→3, 5→4
        num_classes = 5

        print(f"Label-Verteilung:")
        for i in range(num_classes):
            count = np.sum(processed_labels == i)
            original_label = i + 1
            print(f"  Label {original_label}: {count} Samples")

        return processed_labels, num_classes

    def split_data(texts, self, labels, train_ratio=0.8):
        """
        Teilt Daten mit nativen TensorFlow Methoden
        """
        dataset = tf.data.Dataset.from_tensor_slices((texts, labels))
        dataset = dataset.shuffle(buffer_size=len(texts), seed=42)

        total_size = len(texts)
        train_size = int(total_size * train_ratio)

        train_dataset = dataset.take(train_size)
        test_dataset = dataset.skip(train_size)

        print(f"Training samples: {train_size}")
        print(f"Test samples: {total_size - train_size}")

        return train_dataset, test_dataset

    def split_data_tensorflow(self, texts, labels, train_ratio=0.8):
        """
        Teilt Daten mit nativen TensorFlow Methoden
        """
        dataset = tf.data.Dataset.from_tensor_slices((texts, labels))
        dataset = dataset.shuffle(buffer_size=len(texts), seed=42)

        total_size = len(texts)
        train_size = int(total_size * train_ratio)

        train_dataset = dataset.take(train_size)
        test_dataset = dataset.skip(train_size)

        print(f"Training samples: {train_size}")
        print(f"Test samples: {total_size - train_size}")

        return train_dataset, test_dataset

    def train_text_classifier(self, texts, labels, train_ratio=0.8):
        """
        Hauptmethode für Training - ersetzt die alte train_text_classifier
        """
        if texts is None or len(texts) == 0:
            raise ValueError("texts must be a non-empty list of strings")

        # Daten aufteilen
        texts = [str(t) for t in texts]
        labels = [int(l) for l in labels]

        processed_labels, num_classes = self.prepare_labels(labels)

        train_dataset, test_dataset = self.split_data_tensorflow(texts, processed_labels, train_ratio)

        # Tokenizer erstellen
        tokenizer = self.tokenize_data(texts=texts)

        # Preprocessing
        def preprocess_function(text, label):
            return tokenizer(text), label

        train_dataset = train_dataset.map(preprocess_function).batch(32)
        test_dataset = test_dataset.map(preprocess_function).batch(32)

        inputs = layers.Input(shape=(None, ), dtype='int32', name='text_input')
        x = layers.Embedding(tokenizer.vocabulary_size(), 64, mask_zero=True)(inputs)

        x = layers.LSTM(128, return_sequences=True, dropout=0.2)(x)  # Erste LSTM Schicht
        x = layers.LSTM(64, return_sequences=True, dropout=0.2)(x)  # Zweite LSTM Schicht
        x = layers.LSTM(32, return_sequences=True, dropout=0.2)(x)                          # Dritte LSTM Schicht

        attention_output = layers.MultiHeadAttention(
            num_heads=4,  # 4 verschiedene Attention-Mechanismen
            key_dim=16,  # Größe der Attention-Keys
            dropout=0.1
        )(x, x)

        x = layers.GlobalAveragePooling1D()(attention_output)

        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)

        outputs = layers.Dense(5, activation='softmax')(x)


        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        summary = model.summary()

        history = model.fit(
            train_dataset,
            validation_data=test_dataset,
            epochs=20,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )

        return model, tokenizer, test_dataset, summary, history

    def evaluate_model(self, model, test_dataset):
        """
        Evaluiert Model auf test_dataset
        """
        test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        return test_loss, test_accuracy

    def make_predictions(self, model, test_dataset, show_samples=5):
        """
        Macht Predictions auf test_dataset und zeigt Beispiele
        """
        # Komplette Predictions
        all_predictions = model.predict(test_dataset, verbose=0)
        predicted_classes = np.argmax(all_predictions, axis=1)

        preficted_original = predicted_classes + 1  # 0→1, 1→2, 2→3, 3→4, 4→5

        # Zeige einige Beispiele
        sample_count = 0
        for batch in test_dataset:
            if sample_count >= show_samples:
                break

            texts, true_labels = batch
            batch_predictions = model.predict(texts, verbose=0)
            batch_predicted = np.argmax(batch_predictions, axis=1)

            for i in range(min(len(texts), show_samples - sample_count)):
                print(f"\nSample {sample_count + 1}:")
                print(f"True label: {true_labels[i].numpy()}")
                print(f"Predicted label: {batch_predicted[i]}")
                print(f"Confidence: {batch_predictions[i][batch_predicted[i]]:.4f}")
                print("-" * 30)
                sample_count += 1

        return all_predictions, predicted_classes


if __name__ == "__main__":
    process_data_obj = ProcessData()
    collum1 = process_data_obj.load_data(data_cleanup.main().output_file_name, 1)
    collum3 = process_data_obj.load_data(data_cleanup.main().output_file_name, 3)
    #training
    model, tokenizer, test_dataset, summary, history = process_data_obj.train_text_classifier(collum1, collum3)

    #evaluation
    process_data_obj.evaluate_model(model, test_dataset)

    #prdictions auf Test-Set
    predictions, predicted_classes = process_data_obj.make_predictions(model, test_dataset, show_samples=5)

    df = pd.DataFrame(data=predictions, columns=["Predicted Label", ])
    df.to_csv(f"{process_data_obj.exec_path}/output/predictions.csv", index=False)
    print("Predictions saved to output/predictions.csv")

    #print(collum3)
    #print(ProcessData.tokenize_data(texts=collum1))
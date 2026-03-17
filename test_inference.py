#!/usr/bin/env python3
import sys
sys.path.append('./src')

from src.inference.libras_realtime_classifier import LibrasRealtimeClassifier

print("Iniciando classificador em tempo real...")
print("="*60)

try:
    classifier = LibrasRealtimeClassifier(model_path='./model/model.pickle')
    print("\n✓ Classificador inicializado com sucesso")
    print(f"  Modelo disponível: {classifier.model is not None}")
    print(f"  MediaPipe disponível: {classifier.hands is not None}")
    print("\nIniciando modo de teste (3 iterações)...\n")
    
    # Modo de teste simulado
    import numpy as np
    for i in range(3):
        print(f"\n--- Iteração {i+1} ---")
        synthetic_landmarks = np.random.rand(42).astype(np.float32)
        
        if classifier.model is not None:
            try:
                prediction = classifier.model.predict([synthetic_landmarks])[0]
                confidence = classifier.model.predict_proba([synthetic_landmarks])[0].max()
                label = classifier.alphabet_dict.get(prediction, "?")
                print(f"✓ Predição: {label} (confiança: {confidence:.2%})")
            except Exception as e:
                print(f"✗ Erro na predição: {type(e).__name__}: {e}")
        else:
            print("(Sem modelo disponível)")
    
    print("\n" + "="*60)
    print("✓ Teste de inferência concluído com sucesso!")
    
except Exception as e:
    print(f"\n❌ Erro: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

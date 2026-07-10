#!/usr/bin/env python
"""CP4 (b) — prove whether the WASM mel DSP (preemph 0.97 + zero-pad) matches the
Conformer preprocessor.ts, vs a no-preemph + reflect-pad variant. Decisive test:
run each mel through encoder.onnx + ctc_decoder.onnx and see which yields म सचै छु.
"""
import os, json, wave, numpy as np, torch, torchaudio, librosa
import onnxruntime as ort
from huggingface_hub import snapshot_download

REF="म सचै छु"; BLANK=256
A=os.path.join(snapshot_download("ai4bharat/indic-conformer-600m-multilingual"),"assets")
SR,NFFT,WIN,HOP,NMELS=16000,512,400,160,80

# ---- audio (same sinc path as CP1/CP3) ----
with wave.open("test_nepali.wav","rb") as w:
    ch,sw,sr=w.getnchannels(),w.getsampwidth(),w.getframerate(); raw=w.readframes(w.getnframes())
pcm=np.frombuffer(raw,dtype=np.int16).astype(np.float32)/32768.0
if ch>1: pcm=pcm.reshape(-1,ch).mean(1)
wav=torch.from_numpy(pcm).unsqueeze(0)
if sr!=SR: wav=torchaudio.functional.resample(wav,sr,SR)
x0=wav.numpy()[0].astype(np.float32)

# ---- reference mel from preprocessor.ts ----
pp=torch.jit.load(f"{A}/preprocessor.ts",map_location="cpu")
ref_feat,ref_len=pp(input_signal=torch.from_numpy(x0).unsqueeze(0),length=torch.tensor([len(x0)]))
ref_feat=ref_feat.numpy()[0]   # [80, T]

# ---- shared building blocks matching mel_wasm.cpp ----
window=torch.hann_window(WIN,periodic=False).numpy().astype(np.float32)
fb=librosa.filters.mel(sr=SR,n_fft=NFFT,n_mels=NMELS,fmin=0,fmax=SR/2,norm="slaney").astype(np.float32)
win_full=np.zeros(NFFT,np.float32); win_full[(NFFT-WIN)//2:(NFFT-WIN)//2+WIN]=window

def wasm_mel(x, preemph, pad_mode):
    x=x.copy()
    if preemph:                                 # right-to-left, as in the C++
        for i in range(len(x)-1,0,-1): x[i]=x[i]-0.97*x[i-1]
    pad=NFFT//2
    xp=np.pad(x,(pad,pad),mode=pad_mode)
    T=(len(xp)-NFFT)//HOP+1
    power=np.empty((NFFT//2+1,T),np.float32)
    for t in range(T):
        fr=xp[t*HOP:t*HOP+NFFT]*win_full
        S=np.fft.rfft(fr); power[:,t]=(S.real**2+S.imag**2)
    mel=fb@power
    mel=np.log(mel+2.0**-24)
    seq=(len(x)+2*(NFFT//2)-NFFT)//HOP                # seq_len per C++
    out=np.empty_like(mel)
    for m in range(NMELS):
        row=mel[m]; mean=row[:seq].mean()
        sd=np.sqrt(((row[:seq]-mean)**2).sum()/(seq-1))+1e-5
        out[m]=(row-mean)/sd
    return out[:, :seq]

def transcribe(feat):
    feat=np.ascontiguousarray(feat[None].astype(np.float32))
    enc=ort.InferenceSession(f"{A}/encoder.onnx",providers=["CPUExecutionProvider"])
    eo,el=enc.run(["outputs","encoded_lengths"],{"audio_signal":feat,"length":np.array([feat.shape[-1]],np.int64)})
    ctc=ort.InferenceSession(f"{A}/ctc_decoder.onnx",providers=["CPUExecutionProvider"])
    lp=ctc.run(["logprobs"],{"encoder_output":eo})[0]
    mask=np.array(json.load(open(f"{A}/language_masks.json"))["ne"],bool)
    vocab=json.load(open(f"{A}/vocab.json"))["ne"]
    idx=lp[0][:,mask].argmax(-1)
    col=idx[np.insert(np.diff(idx)!=0,0,True)]
    return "".join(vocab[i] for i in col if i!=BLANK).replace("▁"," ").strip()

melA=wasm_mel(x0,preemph=True, pad_mode="constant")   # current WASM
melB=wasm_mel(x0,preemph=False,pad_mode="reflect")    # conformer-correct

def cmp(tag,mel):
    T=min(mel.shape[1],ref_feat.shape[1])
    d=np.abs(mel[:,:T]-ref_feat[:,:T])
    print(f"  {tag}: shape={mel.shape} vs ref{ref_feat.shape}  meanAbsDiff={d.mean():.4f} maxAbsDiff={d.max():.4f}")

print("reference (preprocessor.ts) mel:",ref_feat.shape)
print("--- numeric closeness to reference ---")
cmp("A preemph+zeropad (current WASM)",melA)
cmp("B no-preemph+reflect (conformer)",melB)
print("--- decisive: transcription through encoder+ctc ---")
print("  reference        :",repr(REF))
print("  A (current WASM) :",repr(transcribe(melA)))
print("  B (no-preemph)   :",repr(transcribe(melB)))

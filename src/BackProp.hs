module BackProp
  ( Forward
  , Backward
  , Gradient
  , Layer (..)
  , forward
  , backward
  , softmaxForward
  , softmaxBackward
  , LayerOutput(..)) where


import Prelude hiding (map, zipWith)
import Data.Array.Repa hiding ((++))
import Data.Array.Repa.Algorithms.Matrix

import Functions (softmax, crossEntropyError)


data Forward

data Backward

data Gradient

type Matrix2DU = Array U DIM2 Double

type Matrix2DD = Array D DIM2 Double

type Input = Matrix2DD

type Output = Matrix2DD

data LayerOutput = OutputMatrix Matrix2DD
                 | OutputDouble Double

instance Show LayerOutput where
  show (OutputMatrix _) = "LayerOutput (Array D DIM2 Double)"
  show (OutputDouble d) = "LayerOutput " ++ show d

type Weight = Matrix2DU

type Bias = Matrix2DU

type Dout = Matrix2DD

data Layer a = Affine Weight Bias Input
             | Sigmonoid Matrix2DD
             | ReLU Matrix2DD
             | SoftmaxWithLoss Matrix2DU Matrix2DU
             | None

forward :: Input -> Layer Forward -> (LayerOutput, Layer Backward)
forward i (ReLU _)              = reluForward i
forward i (Sigmonoid _)         = sigmonoidForward i
forward i (Affine w b _)        = affineForward i w b
forward i (SoftmaxWithLoss t _) = softmaxForward i t
forward _ None                  = error "Invalid Layer None"

reluForward :: Input -> (LayerOutput, Layer Backward)
reluForward i = (OutputMatrix out, ReLU i)
  where
    out  = map (\x -> if x <= 0 then 0 else x) i

sigmonoidForward :: Input -> (LayerOutput, Layer Backward)
sigmonoidForward i = (OutputMatrix out, Sigmonoid out)
  where out = map f i
        f x = 1 / (1 + exp(-x))

affineForward :: Input -> Weight -> Bias ->
                 (LayerOutput, Layer Backward)
affineForward i w b = (OutputMatrix (delay x), Affine w b x)
  where i' = computeS i
        x = (mmultS i' w) +^ b

softmaxForward :: Matrix2DD -> Matrix2DU ->
                  (LayerOutput, Layer Backward)
softmaxForward x t = (OutputDouble loss, SoftmaxWithLoss y t)
  where r = row . extent $ x
        y = softmax x
        loss = crossEntropyError y t r

backward :: Dout -> Layer Backward -> (Dout, Layer Gradient)
backward dout (ReLU i)              = reluBackward dout i
backward dout (Sigmonoid i)         = sigmonoidBackward dout i
backward dout (Affine w _ x)        = affineBackward dout w x
backward _    (SoftmaxWithLoss y t) = softmaxBackward y t
backward _    None                  = error "Invalid Layer None"

reluBackward :: Dout -> Matrix2DD -> (Dout, Layer Gradient)
reluBackward ds is = (zipWith f ds is, None)
  where f d i = if i <= 0 then 0 else d

sigmonoidBackward :: Dout -> Matrix2DD -> (Dout, Layer Gradient)
sigmonoidBackward ds is = let dx = ds *^ (map (1.0-) is) *^ is
                          in (dx, None)

affineBackward :: Dout -> Matrix2DU -> Matrix2DD ->
                  (Dout, Layer Gradient)
affineBackward dout w x = (delay dx, Affine dw db undefined)
  where ds = computeS dout
        dx = mmultS ds (ts w)
        dw = mmultS (ts (computeS x)) ds
        db = computeS $ mapSum dout
        mapSum = reshape (ix2 1 (size.extent $ dout)). foldS (+) 0
        ts = computeS . transpose

softmaxBackward :: Matrix2DU -> Matrix2DU -> (Dout, Layer Gradient)
softmaxBackward y t = (dx, None)
  where batsize = row . extent $ y
        dx = map (/(fromIntegral batsize)) (y -^ t)

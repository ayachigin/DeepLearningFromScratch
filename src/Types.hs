module Types where


import Prelude hiding (map, zipWith)
import Data.Array.Repa
import Data.Array.Repa.Algorithms.Matrix


data Forward

data Backward

data Gradient

type Input = Matrix2DD

type Output = Matrix2DD

type Weight = Matrix2DU

type Bias = Matrix2DU

type Dout = Matrix2DD

type Matrix2DU = Array U DIM2 Double

type Matrix2DD = Array D DIM2 Double

data Layer a = Affine Weight Bias Input
             | Sigmonoid Matrix2DD
             | ReLU Matrix2DD
             | None

forward :: Input -> Layer Forward -> (Input, Layer Backward)
forward i (ReLU _)              = reluForward i
forward i (Sigmonoid _)         = sigmonoidForward i
forward i (Affine w b _)        = affineForward i w b
forward _ None                  = error "None is not valid layer"

reluForward :: Input -> (Output, Layer Backward)
reluForward i = (out, ReLU i)
  where
    out  = map (\x -> if x <= 0 then 0 else x) i

sigmonoidForward :: Input -> (Output, Layer Backward)
sigmonoidForward i = (out, Sigmonoid out)
  where out = map f i
        f x = 1 / (1 + exp(-x))

affineForward :: Input -> Weight -> Bias -> (Output, Layer Backward)
affineForward i w b = (delay x, Affine w b x)
  where i' = computeS i
        x = (mmultS i' w) +^ b

softmaxForward :: Matrix2DD -> Matrix2DD -> (Double, Matrix2DD)
softmaxForward x t = (loss, y)
  where y = softmax x
        loss = crossEntropyError y t
        softmax = undefined
        crossEntropyError = undefined

backward :: Dout -> Layer Backward -> (Dout, Layer Gradient)
backward dout (ReLU i)       = reluBackward dout i
backward dout (Sigmonoid i)  = sigmonoidBackward dout i
backward dout (Affine w _ x) = affineBackward dout w x
backward _     None           = error "None is not valid layer"

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

softmaxBackward :: Matrix2DD -> Matrix2DD -> Dout
softmaxBackward x t = dx
  where batsize = row . extent $ x
        dx = map (/(fromIntegral batsize)) (x -^ t)

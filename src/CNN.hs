{-# LANGUAGE TypeOperators, TypeSynonymInstances, FlexibleInstances #-}
module CNN where

import Prelude hiding (map, (++), zipWith, traverse)
import Data.Array.Repa

import Types

data Forward

data Backward

type MiniBatchInput = Array D DIM4 Double

data Convolution = Convolution (Matrix D) (Matrix D) Int Int

data CNNLayer a = ConvolutionLayer Convolution
                | Pooling

data Im2ColOption = Im2ColOption
  { filterWidth  :: Int
  , filterHeight :: Int
  , strideSize   :: Int
  , padSize      :: Int
  } deriving (Show, Read, Eq)

class Pad a where
  pad :: (Source r e, Num e) =>
         Array r a e -> Int -> Array D a e

instance Pad DIM2 where
  pad = pad2d

instance Pad DIM3 where
  pad = pad3d

{- | pad2d
>>> x = fromListUnboxed (ix2 2 3) [(1::Int)..6]
>>> computeS $ pad2d x 1
AUnboxed ((Z :. 4) :. 5) [0.0,0.0,0.0,0.0,0.0,0.0,1.0,2.0,3.0,0.0,0.0,4.0,5.0,6.0,0.0,0.0,0.0,0.0,0.0,0.0]
-}
pad2d :: (Source r1 e, Num e) =>
       Array r1 DIM2 e -> Int -> Array D DIM2 e
pad2d arr p = fromFunction (ix2 (x+2*p) (y+2*p)) f
  where (Z:.x:.y) = extent arr
        f (Z:.x':.y')= if g then 0
                       else index arr (ix2 (x'-p) (y'-p))
          where g = x' < p || y' < p || x' >= x + p || y' >= y + p

pad3d :: (Source r e, Num e) =>
         Array r DIM3 e -> Int -> Array D DIM3 e
pad3d arr p = fromFunction (ix3 ch (x+2*p) (y+2*p)) f
      where
        (Z:.ch:.x:.y) = extent arr
        f (Z:.ch':.x':.y')
          | isPad     = 0
          | otherwise = index arr (ix3 ch' (x'-p) (y'-p))
          where isPad = x' < p || y' < p || x' >= x + p || y' >= y + p

class Im2Col shape where
  im2col :: (Source r e, Num e) =>
            Im2ColOption ->
            Array r shape e ->
            Array D DIM2 e

instance Im2Col DIM3 where im2col = im2col3d

instance Im2Col DIM4 where im2col = im2col4d

im2col4d :: (Source r e, Num e) =>
            Im2ColOption ->
            Array r DIM4 e ->
            Array D DIM2 e
im2col4d opts arr = reshape (ix2 colY colX) col
  where
    (Z:.colX:.colY) = extent col
    col = foldl1 (++) newArrs
    newArrs = [im2col3d opts $ fromFunction (ix3 x y z) (g w')
              | w' <- [0..w-1]]
      where g w' (Z:.x':.y':.z') = index arr (ix4 w' x' y' z')
    (Z:.w:.x:.y:.z) = extent arr

im2col3d :: (Source r e, Num e) =>
          Im2ColOption ->
          Array r DIM3 e ->
          Array D DIM2 e
im2col3d (Im2ColOption fw fh st p) arr = fromFunction newShape f
  where
    f (Z:.x':.y') = index padarr (ix3 a b c)
      where a = (y' `div` (fw*fh)) `rem` ch
            b = y' `div` fw - (a * fw) + (x' `div` ax)
            c = (y' - (y' `div` fw * fw)) + (x' `rem` ax)
            ax = x `div` st - (fw - st)
            ay = y `div` st - (fh - st)
    newShape = (ix2 (x'*y') (fw * fh * ch))
      where x' = (x `div` st) - (fw - st)
            y' = (y `div` st) - (fh - st)
    (Z:.ch:.x:.y) = extent padarr
    padarr = pad3d arr p


convolutionForward :: MiniBatchInput ->
                      Convolution ->
                      CNNLayer Backward
convolutionForward = undefined

-- | focus
--
-- It provides access to the array that reduced one dimension of
-- muitidimensional array by index.
focus :: (Source r e, Shape sh) =>
         (Array r (sh :. Int) e) ->
         Int ->
         Array D sh e
focus arr idx = fromFunction sh' f
  where (sh' :. _) = extent arr
        f sh = index arr (sh :. idx)


test :: Array U DIM2 Float
test = computeS $ pad 2 x
  where x = fromListUnboxed (ix2 2 3) [(1::Float)..6]

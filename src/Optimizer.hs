module Optimizer where

import Prelude hiding (map)
import Data.Array.Repa hiding (zipWith)

import Types

type Momentum = [(Weight, Bias)]

moment :: Double
moment = 0.9

updateMomentumMaybe :: Double -> Double -> Maybe Momentum -> NN -> Gradients ->
                       (NN, Momentum)
updateMomentumMaybe m lr vs n gs = case vs of
                                     (Just vs') ->
                                       updateMomentum m lr vs' n gs
                                     Nothing ->
                                       updateMomentum m lr evs n gs
  where
    evs = fmap f n
    f (w, b, _) = ( computeS $ map (const 0) w
                  , computeS $ map (const 0) b)

updateMomentum :: Double -> Double -> Momentum -> NN -> Gradients ->
                  (NN, Momentum)
updateMomentum m lr vs n gs = (newNN, m')
  where
    newNN = zipWith f n m'
    f (w, b, fn) (mw, mb) = ( computeS $ w +^ mw
                           , computeS $ b +^ mb
                           , fn)
    m' = zipWith g vs gs
    g :: (Weight, Bias) -> ([Double], [Double]) -> (Weight, Bias)
    g (vw, vb) (gw, gb) = ( computeS $ (map (*m) vw) -^ (map (*lr) gwa)
                          , computeS $ (map (*m) vb) -^ (map (*lr) gba))
      where
        gwa = fromListUnboxed (extent vw) gw
        gba = fromListUnboxed (extent vb) gb

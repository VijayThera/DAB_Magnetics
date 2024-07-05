Include "Parameter.pro";
Function{
  b = {np.float64(0.0), np.float64(0.14285714285714285), np.float64(0.2857142857142857), np.float64(0.42857142857142855), np.float64(0.5714285714285714), np.float64(0.7142857142857142), np.float64(0.8571428571428571), np.float64(1.0)} ;
  mu_real = {np.float64(3000.0), np.float64(2981.632857142857), np.float64(2937.4157142857143), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0), np.float64(1.0)} ;
  mu_imag = {np.float64(1.0), np.float64(321.68142857142857), np.float64(604.3842857142856), np.float64(636.25), np.float64(636.25), np.float64(636.25), np.float64(636.25), np.float64(636.25)} ;
  mu_imag_couples = ListAlt[b(), mu_imag()] ;
  mu_real_couples = ListAlt[b(), mu_real()] ;
  f_mu_imag_d[] = InterpolationLinear[Norm[$1]]{List[mu_imag_couples]};
  f_mu_real_d[] = InterpolationLinear[Norm[$1]]{List[mu_real_couples]};
  f_mu_imag[] = f_mu_imag_d[$1];
  f_mu_real[] = f_mu_real_d[$1];
 }  
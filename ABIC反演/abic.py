import numpy as np
import pandas as pd
import regula

def minimize_datamisfit(gobs, kernel, ref_den, shape):
    
    def min_abic(x):
                #构建光滑项
        Lx = regula.Lx(shape)
        #print("Lx shape:", Lx.shape)
        Ly = regula.Ly(shape)
        #print("Ly shape:", Ly.shape)
        Lz = regula.Lz(shape)
        #print("Ly shape:", Lz.shape)

        # x方向一阶光滑强度
        alpha = np.exp(x[0])
        # y方向一阶光滑强度
        beta = np.exp(x[1])
        # z方向一阶光滑强度
        zeta = np.exp(x[2])
        # 数据可信度
        Cd_c = np.exp(x[3])
        # 模型可信度
        Cp_c = np.exp(x[4])

        norm = alpha*np.dot(Lx.T, Lx) + beta*np.dot(Ly.T, Ly) + zeta*np.dot(Lz.T, Lz)

        ## 定义参考模型
        Cd = np.eye(gobs.shape[0])/Cd_c
        #print("Cd shape:", Cd.shape)
        Cd_inv = np.linalg.pinv(Cd)
        #print("Cd_inv shape:", Cd_inv.shape)

        ref_density = ref_den.ravel()
        ref_density = ref_density.reshape(512, 1)
        Cp = np.eye(512)/Cp_c
        Cp_inv = np.linalg.pinv(Cp)

        #print("Cp_inv shape:", Cp_inv.shape)
        #print("Ref density shape:", ref_density.shape)
        #print("Kernal shape:", kernel.shape)
        #print("G_obs shape:", gobs.shape)

        m_lse1 = np.linalg.pinv(np.dot(np.dot(kernel.T, Cd_inv), kernel) + Cp_inv + norm)
        #print("m_lse1 shape:", m_lse1.shape)
        m_lse2 = np.dot(np.dot(kernel.T, Cd_inv), gobs) + np.dot(Cp_inv, ref_density)
        #print("m_lse2 shape:", m_lse2.shape)
        result = np.dot(m_lse1, m_lse2)
        res_lse = np.array(result).reshape(8, 8, 8)
        #print("Result shape:", res_lse.shape)
        predict = np.mat(kernel)*np.transpose(np.mat(res_lse.ravel()))
        #print("Predict shape:", predict.shape)
        datamisfit = np.linalg.norm(predict-gobs, ord=1)

        return datamisfit
    
    return min_abic

def minimize_abic(gobs, kernel, ref_den, shape, Nh=5):
    
    def min_abic(x):
                #构建光滑项
        Lx = regula.Lx(shape)
        #print("Lx shape:", Lx.shape)
        Ly = regula.Ly(shape)
        #print("Ly shape:", Ly.shape)
        Lz = regula.Lz(shape)
        #print("Ly shape:", Lz.shape)

        # x方向一阶光滑强度
        alpha = np.exp(x[0])+1e-9
        # y方向一阶光滑强度
        beta = np.exp(x[1])+1e-9
        # z方向一阶光滑强度
        zeta = np.exp(x[2])+1e-9
        # 数据可信度
        Cd_c = np.exp(x[3])+1e-9
        # 模型可信度
        Cp_c = np.exp(x[4])+1e-9

        norm = np.dot(Lx.T, Lx)/alpha + np.dot(Ly.T, Ly)/beta + np.dot(Lz.T, Lz)/zeta

        ## 定义参考模型
        Cd = np.eye(gobs.shape[0])/Cd_c
        #print("Cd shape:", Cd.shape)
        Cd_inv = np.linalg.pinv(Cd)
        #print("Cd_inv shape:", Cd_inv.shape)

        ref_density = ref_den.ravel()
        ref_density = ref_density.reshape(512, 1)
        Cp = np.eye(512)/Cp_c
        Cp_inv = np.linalg.pinv(Cp)

        #print("Cp_inv shape:", Cp_inv.shape)
        #print("Ref density shape:", ref_density.shape)
        #print("Kernal shape:", kernel.shape)
        #print("G_obs shape:", gobs.shape)

        m_lse1 = np.linalg.pinv(np.dot(np.dot(kernel.T, Cd_inv), kernel) + Cp_inv + norm)
        #print("m_lse1 shape:", m_lse1.shape)
        m_lse2 = np.dot(np.dot(kernel.T, Cd_inv), gobs) + np.dot(Cp_inv, ref_density)
        #print("m_lse2 shape:", m_lse2.shape)
        result = np.dot(m_lse1, m_lse2)
        res_lse = np.array(result).reshape(8, 8, 8)
        #print("Result shape:", res_lse.shape)
        predict = np.mat(kernel)*np.transpose(np.mat(res_lse.ravel()))
        #print("Predict shape:", predict.shape)
        datamisfit = np.linalg.norm(predict-gobs, ord=1)

        abic = (np.log(2*np.pi*alpha) + np.log(2*np.pi*beta)+
                np.log(2*np.pi*zeta) + np.log(2*np.pi*Cd_c) + np.log(2*np.pi*Cp_c))
        #print('x: ', x)
        
        final_misfit = datamisfit + abic
        
        with open('result.txt', 'a') as f:
            for xi in x:
                f.write(str(xi))
                f.write('  ')
            f.write(str(final_misfit))
            f.write('\n')
        
        return final_misfit
    return min_abic
    
import copy
import numpy as np
import itertools
import torch
from torch.autograd import Variable
from torch import optim
from IPython.display import clear_output

from meta_optimizer import Optimizer_LSTM, Optimizer_HNN
from helpers import *

def one_step_fit_LSTM(opt_net, meta_opt, target_cls, target_to_opt, \
                      unroll, optim_it, n_epochs, out_mul, should_train=True):
    if should_train:
        opt_net.train()
    else:
        opt_net.eval()
        unroll = 1
    
    target = target_cls(training=should_train)
    optimizee = w(target_to_opt())
    n_params = 0
    for name, p in optimizee.all_named_parameters():
        n_params += int(np.prod(p.size()))
    hidden_states = [w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
    cell_states = [w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
    all_losses_ever = []
    if should_train:
        meta_opt.zero_grad()
    all_losses = None
    for iteration in range(1, optim_it + 1):
        loss = optimizee(target)
                    
        if all_losses is None:
            all_losses = loss
        else:
            all_losses += loss
        
        all_losses_ever.append(loss.data.cpu().numpy())
        loss.backward(retain_graph=should_train)

        offset = 0
        result_params = {}
        hidden_states2 = [w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
        cell_states2 = [w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
        for name, p in optimizee.all_named_parameters():
            cur_sz = int(np.prod(p.size()))
            # We do this so the gradients are disconnected from the graph but we still get
            # gradients from the rest
            gradients = detach_var(p.grad.view(cur_sz, 1))
            updates, new_hidden, new_cell = opt_net(
                gradients,
                [h[offset:offset+cur_sz] for h in hidden_states],
                [c[offset:offset+cur_sz] for c in cell_states]
            )
            for i in range(len(new_hidden)):
                hidden_states2[i][offset:offset+cur_sz] = new_hidden[i]
                cell_states2[i][offset:offset+cur_sz] = new_cell[i]
            result_params[name] = p + updates.view(*p.size()) * out_mul
            result_params[name].retain_grad()
            
            offset += cur_sz
            
        if iteration % unroll == 0:
            if should_train:
                meta_opt.zero_grad()
                all_losses.backward()
                meta_opt.step()
                
            all_losses = None

            optimizee = w(target_to_opt())
            optimizee.load_state_dict(result_params)
            optimizee.zero_grad()
            hidden_states = [detach_var(v) for v in hidden_states2]
            cell_states = [detach_var(v) for v in cell_states2]
            
        else:
            for name, p in optimizee.all_named_parameters():
                rsetattr(optimizee, name, result_params[name])
            assert len(list(optimizee.all_named_parameters()))
            hidden_states = hidden_states2
            cell_states = cell_states2
    return all_losses_ever

def one_step_fit_HNN(opt_net, meta_opt, target_cls, target_to_opt, \
                     unroll, optim_it, n_epochs, out_mul, should_train=True):
    if should_train:
        opt_net.train()
    else:
        opt_net.eval()
        unroll = 1
    
    target = target_cls(training=should_train)
    optimizee = w(target_to_opt())
    n_params = 0
    for name, p in optimizee.all_named_parameters():
        n_params += int(np.prod(p.size()))
    #hidden_states = [w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
    derivative_input = w(Variable(torch.zeros(n_params, 1)))
    all_losses_ever = []
    if should_train:
        meta_opt.zero_grad()
    all_losses = None
    for iteration in range(1, optim_it + 1):
        loss = optimizee(target)
        # print(iteration)      
        if all_losses is None:
            all_losses = loss
        else:
            all_losses += loss
        
        all_losses_ever.append(loss.data.cpu().numpy())
        loss.backward(retain_graph=should_train)

        offset = 0
        result_params = {}
        derivative_input2 = w(Variable(torch.zeros(n_params, 1)))
        for name, p in optimizee.all_named_parameters():
            cur_sz = int(np.prod(p.size()))
            # We do this so the gradients are disconnected from the graph but we still get
            # gradients from the rest
            gradients = detach_var(p.grad.view(cur_sz, 1))
            # print(p.shape, gradients.shape)
            #norm = torch.norm(gradients)
            inp = torch.cat((gradients, derivative_input[offset:offset+cur_sz]), dim=-1)
            updates = opt_net(inp)
            #derivative_input2[offset:offset+cur_sz, 0] = (updates.view(*p.size())).reshape(-1,)* out_mul
            derivative_input2[offset:offset+cur_sz, 0] = (updates.view(*p.size())).reshape(-1,) \
                                                         / torch.norm((updates.view(*p.size())).reshape(-1,))
            result_params[name] = p + updates.view(*p.size())*out_mul
            result_params[name].retain_grad()
            offset += cur_sz
            
        if iteration % unroll == 0:
            if should_train:
                meta_opt.zero_grad()
                all_losses.backward()
                meta_opt.step()
                
            all_losses = None

            optimizee = w(target_to_opt())
            optimizee.load_state_dict(result_params)
            optimizee.zero_grad()
            derivative_input = detach_var(derivative_input2)
        else:
            for name, p in optimizee.all_named_parameters():
                rsetattr(optimizee, name, result_params[name])
            assert len(list(optimizee.all_named_parameters()))
            derivative_input = derivative_input2

    return all_losses_ever


def fit(target_cls, target_to_opt, model_name, preproc=False, unroll=20, \
        optim_it=100, n_epochs=20, n_tests=100, lr=0.001, out_mul=1.0):
    assert (model_name == 'HNN') or (model_name == 'LSTM')
    
    if model_name == 'LSTM':
        opt_net = w(Optimizer_LSTM(preproc=preproc))
    else:
        opt_net = w(Optimizer_HNN(preproc=preproc))
        
    meta_opt = optim.Adam(opt_net.parameters(), lr=lr)
    best_net = None
    best_loss = np.inf
    
    for _ in range(n_epochs):
        for _ in range(20):
            if model_name == 'LSTM':
                one_step_fit_LSTM(opt_net, meta_opt, target_cls, target_to_opt, \
                                  unroll, optim_it, n_epochs, out_mul, should_train=True)
                loss = np.mean([np.mean(one_step_fit_LSTM(opt_net, meta_opt, target_cls, target_to_opt, \
                                      unroll, optim_it, n_epochs, out_mul, should_train=False)) for _ in range(n_tests)])
            else:
                one_step_fit_HNN(opt_net, meta_opt, target_cls, target_to_opt,\
                                 unroll, optim_it, n_epochs, out_mul, should_train=True)
                loss = np.mean([np.mean(one_step_fit_HNN(opt_net, meta_opt, target_cls, target_to_opt,\
                                      unroll, optim_it, n_epochs, out_mul, should_train=False)) for _ in range(n_tests)])
        
        if loss < best_loss:
            best_loss = loss
            best_net = copy.deepcopy(opt_net.state_dict())
    return best_loss, best_net

def fit_normal(target_cls, target_to_opt, opt_class, lr, n_tests=100, n_epochs=100, **kwargs):
    results = []
    for i in range(n_tests):
        target = target_cls(training=False)
        optimizee = w(target_to_opt())
        optimizer = opt_class(optimizee.parameters(), lr=lr, **kwargs)
        total_loss = []
        for _ in range(n_epochs):
            loss = optimizee(target)
            
            total_loss.append(loss.data.cpu().numpy())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        results.append(total_loss)
    return results

def find_best_parameters_HNN(target_cls, target_to_opt, preproc=False):
    best_loss = np.inf
    best_lr = 0.0
    best_out_mul = 0.0
    lrs = [0.1, 0.01, 0.001, 0.003, 0.0001]
    out_muls= [0.1, 0.01, 0.001, 0.0001]    
    for lr, out_mul in itertools.product(lrs, out_muls):
        print('Trying:', lr, out_mul)
        loss = best_loss + 1.0
        loss = fit(target_cls, target_to_opt, 'HNN', preproc=preproc, unroll=20, optim_it=100,\
                            n_epochs=20, n_tests=10, lr=lr, out_mul=out_mul)[0]
        if loss < best_loss:
            best_loss = loss
            best_lr = lr
            best_out_mul = out_mul
        print(best_loss, best_lr, best_out_mul)
    clear_output()        
    return best_loss, best_lr, best_out_mul
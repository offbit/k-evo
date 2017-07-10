"""Tournament play experiment."""
from __future__ import absolute_import
import dna
import gp
import cPickle
# Use cuda ?


if __name__=='__main__':
    # setup a tournament!
    nb_evolution_steps = 10
    tournament = \
        gp.TournamentOptimizer(
            population_sz=20,
            init_fn=dna.random_net,
            nb_workers=2,
            use_cuda=True)

    for i in range(nb_evolution_steps):
        print('\nEvolution step:{}'.format(i))
        print('================')
        tournament.step()
        # keep track of the experiment results & corresponding architectures
        name = "tourney_{}".format(i)
        cPickle.dump(tournament.stats, open(name + '.stats','wb'))
        cPickle.dump(tournament.history, open(name +'.pop','wb'))
# Eleflow

Nesse repositório contem as respostas para o processo seletivo da Eleflow Big Data.

## Q1

A questão está contida no notebook video_games_Q1.

## Q2

Está contida nos slides.

## Q3

Na questão 3 foi solucionado o problema para previsão se um filme deve ou não ser adicionado no catálogo, com objetivo de melhoramento do catálogo.
Para isso seriam possíveis duas abordagens a decisão de um valor de corte fixo para nota mínima que esse filme precisa obter para ser adicionado no catálogo ou (a abordagem que utilizei) um valor de corte dinâmico baseado nas notas do catálogo atual.

Utilizando a segunda abordagem eu pensei em 3 casos possíveis dependendo do objetivo de negócio.
Utilizar o primeiro quartil das notas do catálogo atual como nota de corte: com nota de corte mais baixa possibilita um aumento rápido do catálogo, e evita que o catálogo possua filmes com uma avaliação muito baixa. Ideal para um momento de rápida expansão de catálogo.

Utilizar o terceiro quartil das notas como nota de corte: O número possível de filmes é baixo impedindo uma expansão extensa do catálogo, vai auxiliar na melhoria da qualidade do catálogo. Ideal para um momento de manutenção e melhoria do catálogo.

Utilizar o segundo quartil/mediana (Abordagem utilizada): O número possível de filmes a ser licenciado entra em uma faixa intermediária, possibilitando uma expansão do catálogo enquanto gera uma melhora na qualidade do catálogo. Ideal para uma abordagem intermediária entre crescimento e melhoria da qualidade do catálogo.

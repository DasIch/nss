/*
This is an example nss file.

Properties of something are defined like this

selector:
    property: value
    next-property: next-value
*/

base_padding = 2px
background_color = #eee
text_color = #111
link_color = #ff0000

body:
    font-family: serif, sans-serif, 'Verdana', 'Times New Roman'
    color: $text_color
    padding->
        top: $base_padding + 2
	right: $base_padding + 3
	left: $base_padding + 3
	bottom: $base_padding + 2
    background-color: $background_color

div.foo:
    width: "Hello World".length() * 20px
    foo: (foo, bar, baz, 42).join('/')

a:
    color: $link_color
    &:hover:
        color: $link_color.darken(30%)
    &:active:
        color: $link_color.brighten(10%)

div.navigation:
    height: 1.2em
    padding: 0.2em
    ul:
        margin: 0
        padding: 0
        list-style: none
        li:
            float: left
            height: 1.2em
            a:
	        display: block
                height: 1em
                padding: 0.1em
    foo: (1 2 3).string()

package tub.ods.source;

public interface Source<T> {

    public T source();

    public boolean end();
}
